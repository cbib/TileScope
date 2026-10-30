#' Create CSV from a Seurat object for use as TileScope input
#'
#' Builds per-pair CSV files containing spatial coordinates, two
#' expression/score columns, and an ROI label, ready to be consumed by
#' TileScope's `run_tilescope_from_csv()`. The function dispatches on the type
#' of `input_data`:
#'   * character vector           -> all pairwise combinations of the listed genes
#'   * list of length-2 vectors   -> the listed gene pairs, as-is
#'   * named list of gene vectors -> metaprogram signatures scored via AddModuleScore
#'
#' @param seurat_obj           Seurat object (Visium spatial transcriptomics).
#' @param sample_id            Identifier used in output file names.
#' @param assay                Assay to read from (e.g. "Spatial2").
#' @param layer                Layer to read from. Defaults to "data" (the
#'                             expression-scale layer). A value supplied by the
#'                             caller (e.g. from the YAML config) overrides this
#'                             default. NOTE: for the signature path,
#'                             AddModuleScore scores on the assay's `data` layer
#'                             internally regardless of this value; `layer` only
#'                             affects the gene-pair FetchData calls and the
#'                             available-gene lookup (which is layer-invariant).
#'                             To keep gene-pair and signature modes on the same
#'                             layer, leave this at "data" in both config files.
#' @param metadata_col         meta.data column holding the ROI/cluster label.
#' @param input_data           Genes, gene pairs, or named signatures (see above).
#' @param out_path             Output directory (must end with a path separator).
#' @param addmodulescore_nbin  Number of expression bins for AddModuleScore.
#' @param addmodulescore_ctrl  Control genes per bin for AddModuleScore. Must be
#'                             < (n_genes / nbin) or AddModuleScore errors with
#'                             "cannot take a sample larger than the population".
#'
#' @section Externally supplied:
#'   For the named-signature path, `use_selected_mp_pairs` (logical) and
#'   `metaprogram_pairs` (list of length-2 signature-name vectors) are resolved
#'   from the calling environment, not passed as arguments.
#'
#' @return Writes CSV files to `out_path`; invisibly returns NULL.
#' @export
extract_data_from_seurat_to_csv <- function(seurat_obj,
                                            sample_id,
                                            assay,
                                            metadata_col,
                                            input_data,
                                            out_path,
                                            layer               = "data",
                                            addmodulescore_nbin = 24,
                                            addmodulescore_ctrl = 50) { # nolint
  
  # Genes available in the requested assay/layer (row names are layer-invariant)
  assay_genes <- rownames(SeuratObject::LayerData(seurat_obj,
                                                  assay = assay,
                                                  layer = layer))
  
  # Spatial coordinates and ROI labels (shared by all branches)
  coords <- Seurat::GetTissueCoordinates(seurat_obj)
  colnames(coords) <- c("x", "y")
  meta <- seurat_obj@meta.data[[metadata_col]]
  
  # ---------------------------------------------------------------------------
  # Branch 1: input is a flat character vector -> all pairwise combinations
  # ---------------------------------------------------------------------------
  if (is_character_vector(input_data)) {
    
    message("Input is a list of genes")
    
    present_input_data    <- intersect(input_data, assay_genes)
    pairwise_combinations <- t(combn(present_input_data, 2))
    
    for (i in seq_len(nrow(pairwise_combinations))) {
      
      gene1 <- pairwise_combinations[i, 1]
      gene2 <- pairwise_combinations[i, 2]
      pair  <- c(gene1, gene2)
      
      expr <- FetchData(seurat_obj, assay = assay, layer = layer, vars = pair)
      colnames(expr) <- c("gene1", "gene2")
      
      df <- data.frame(
        x = coords[, "x"],
        y = coords[, "y"],
        expr,
        roi_id = meta
      )
      
      filename <- glue::glue("{out_path}{sample_id}_expression_data_{gene1}_{gene2}.csv")
      write.csv(df, filename, row.names = FALSE)
    }
    
    # ---------------------------------------------------------------------------
    # Branch 2: input is a list of length-2 gene pairs -> use as-is
    # ---------------------------------------------------------------------------
  } else if (is_list_of_character_pairs(input_data)) {
    
    message("Input is a list of pairs")
    
    for (pair in input_data) {
      
      gene1 <- pair[1]
      gene2 <- pair[2]
      
      expr <- FetchData(seurat_obj, assay = assay, layer = layer, vars = pair)
      colnames(expr) <- c("gene1", "gene2")
      
      df <- data.frame(
        x = coords[, "x"],
        y = coords[, "y"],
        expr,
        roi_id = meta
      )
      
      filename <- glue::glue("{out_path}{sample_id}_expression_data_{gene1}_{gene2}.csv")
      write.csv(df, filename, row.names = FALSE)
    }
    
    # ---------------------------------------------------------------------------
    # Branch 3: input is a named list of signatures -> AddModuleScore
    # ---------------------------------------------------------------------------
  } else if (is_named_list_of_vectors(input_data)) {
    
    message("Input is a list of signatures")
    
    # Keep only genes present in the assay; drop signatures with no genes left
    filtered_signatures <- list()
    for (sig_name in names(input_data)) {
      
      sig_genes     <- input_data[[sig_name]]
      genes_present <- sig_genes[sig_genes %in% assay_genes]
      genes_missing <- sig_genes[!sig_genes %in% assay_genes]
      
      if (length(genes_present) > 0) {
        filtered_signatures[[sig_name]] <- genes_present
        message("  ", sample_id, " - ", sig_name, ": ",
                length(genes_present), "/", length(sig_genes), " genes present")
        if (length(genes_missing) > 0) {
          cat("Warning - Sample", sample_id, "- Signature", sig_name, ":",
              length(genes_missing), "genes missing:",
              paste(genes_missing, collapse = ", "), "\n")
        }
      } else {
        cat("Warning - Sample", sample_id, "- Signature", sig_name,
            ": No genes found in assay. Skipping this signature.\n")
      }
    }
    
    if (length(filtered_signatures) == 0) {
      stop("No signatures have any genes available in the assay!")
    }
    
    # Score signatures. nbin/ctrl are exposed because the AddModuleScore default
    # ctrl = 100 fails on assays with fewer than ~100 genes per expression bin
    # (bins hold ~ n_genes / nbin genes; control sampling is without replacement).
    seurat_obj <- Seurat::AddModuleScore(
      object   = seurat_obj,
      assay    = assay,
      features = filtered_signatures,
      name     = names(filtered_signatures),
      nbin     = addmodulescore_nbin,
      ctrl     = addmodulescore_ctrl
    )
    
    # AddModuleScore appends an index to each name (name1, name2, ...).
    # Because `name` is the vector of signature names, columns come out as
    # InfImm1, TEM2, ... -> rebuild those names, then relabel to the bare names.
    score_cols <- paste0(names(filtered_signatures), seq_along(filtered_signatures))
    expr <- seurat_obj@meta.data[, score_cols, drop = FALSE]
    colnames(expr) <- names(filtered_signatures)
    
    if (use_selected_mp_pairs) {
      
      # Only signatures that survived filtering exist as columns of `expr`
      available_sigs <- colnames(expr)
      
      for (signature_pair in metaprogram_pairs) {
        
        sig1 <- signature_pair[[1]]
        sig2 <- signature_pair[[2]]
        
        # Guard: skip the pair if either signature was dropped (no genes present).
        # Prevents "undefined columns selected" when indexing `expr`.
        missing_sigs <- setdiff(c(sig1, sig2), available_sigs)
        if (length(missing_sigs) > 0) {
          cat("Warning - Sample", sample_id, "- skipping pair",
              paste(sig1, sig2, sep = " / "),
              ": signature(s) not available:",
              paste(missing_sigs, collapse = ", "), "\n")
          next
        }
        
        sig_expr <- expr[c(sig1, sig2)]
        colnames(sig_expr) <- c("gene1", "gene2")
        
        df <- data.frame(
          x = coords[, "x"],
          y = coords[, "y"],
          sig_expr,
          roi_id = meta
        )
        
        filename <- glue::glue("{out_path}{sample_id}_expression_data_{sig1}_{sig2}.csv")
        write.csv(df, filename, row.names = FALSE)
      }
      
    } else {
      
      # One CSV containing all signature scores as columns
      df <- data.frame(
        x = coords[, "x"],
        y = coords[, "y"],
        expr,
        roi_id = meta
      )
      
      filename <- glue::glue("{out_path}{sample_id}_expression_data.csv")
      write.csv(df, filename, row.names = FALSE)
    }
  }
  
  invisible(NULL)
}


is_character_vector <- function(x) {
  is.character(x) && is.atomic(x)
}


is_list_of_character_pairs <- function(x) {
  is.list(x) &&
    length(x) > 0 &&
    all(sapply(x, function(elem) {
      is.character(elem) && length(elem) == 2
    }))
}


is_named_list_of_vectors <- function(x) {
  if (!is.list(x)) return(FALSE)
  if (length(x) == 0) return(FALSE)
  if (is.null(names(x)) || any(names(x) == "")) return(FALSE)
  
  all(sapply(x, function(elem) {
    is.character(elem) && is.vector(elem) && length(elem) > 0
  }))
}