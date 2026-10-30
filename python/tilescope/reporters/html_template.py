"""HTML report skeleton: header (with all CSS) and footer.

Kept separate from html_summary so the CSS block does not swamp the
section-builder logic. Retained classes: .container, .links and
.link-section. Native tags (table, th, td, details, summary, h1-h3, a)
are styled directly.
"""


def build_html_header() -> str:
    """Return the report's `<!DOCTYPE html>` through opening `<div class="container">`."""
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>TileScope Analysis Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        details {
            margin: 8px 0;
            padding: 8px 12px;
            background-color: #fafbfc;
            border: 1px solid #ecf0f1;
            border-radius: 4px;
        }
        details summary {
            cursor: pointer;
            font-weight: 600;
            color: #34495e;
        }
        .links {
            margin: 10px 0;
        }
        .link-section {
            display: inline-block;
            margin-right: 15px;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
<div class="container">
"""


def build_html_footer() -> str:
    """Return the report's closing `</div></body></html>`."""
    return """
</div>
</body>
</html>
"""
