"""
Build the standalone HTML file by combining the template and JavaScript.

This script reads the standalone template and JavaScript files,
combines them into a single HTML file that can be shared.
"""

import argparse
import os
from pathlib import Path


def build_standalone(template_path: str, js_path: str, output_path: str, summaries_path: str = None) -> None:
    """
    Build the standalone HTML file.

    Args:
        template_path: Path to the HTML template
        js_path: Path to the standalone JavaScript file
        output_path: Path for the output HTML file
        summaries_path: Optional path to example summaries JSON file
    """

    # Read template
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Read JavaScript
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()

    # Read example summaries if provided
    summaries_content = "null"
    if summaries_path and Path(summaries_path).exists():
        with open(summaries_path, "r", encoding="utf-8") as f:
            summaries_content = f.read()

    # Embed summaries data in JavaScript
    js_content_with_summaries = f"""
    // Embedded example summaries data
    window.exampleSummaries = {summaries_content};

    {js_content}
    """

    # Replace placeholder with JavaScript content
    if "// STANDALONE_JS_PLACEHOLDER" in html_content:
        html_content = html_content.replace("// STANDALONE_JS_PLACEHOLDER", js_content_with_summaries)
    else:
        print("Warning: STANDALONE_JS_PLACEHOLDER not found in template")
        # Append JavaScript at the end as fallback
        html_content = html_content.replace(
            "</body>", f"<script>\n{js_content_with_summaries}\n</script>\n</body>"
        )

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Calculate file size
    file_size = os.path.getsize(output_path)
    size_mb = file_size / (1024 * 1024)

    print("Standalone HTML file created successfully!")
    print(f"Output: {output_path}")
    print(f"File size: {size_mb:.2f} MB")

    if size_mb > 10:
        print("\nWarning: The output file is quite large (>10MB).")
        print(
            "Consider using CDN links for Bootstrap and Papa Parse to reduce file size."
        )


def main():
    parser = argparse.ArgumentParser(description="Build standalone HTML viewer")
    parser.add_argument(
        "--template",
        default="templates/standalone_template.html",
        help="Path to HTML template (default: templates/standalone_template.html)",
    )
    parser.add_argument(
        "--javascript",
        default="static/standalone.js",
        help="Path to standalone JavaScript (default: static/standalone.js)",
    )
    parser.add_argument(
        "--output",
        default="standalone.html",
        help="Output path for standalone HTML (default: standalone.html)",
    )
    parser.add_argument(
        "--summaries",
        help="Optional path to example summaries JSON file",
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    template_path = script_dir / args.template
    js_path = script_dir / args.javascript
    output_path = script_dir / args.output
    summaries_path = script_dir / args.summaries if args.summaries else None

    # Check if input files exist
    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}")
        return 1

    if not js_path.exists():
        print(f"Error: JavaScript file not found: {js_path}")
        return 1

    # Build the standalone file
    build_standalone(str(template_path), str(js_path), str(output_path), str(summaries_path) if summaries_path else None)

    print("\n=== Next Steps ===")
    print("1. Run export_standalone.py to create the data CSV and config")
    if summaries_path:
        print("2. Example summaries embedded in standalone file")
        print("3. Share these files with clinicians:")
    else:
        print("2. Share these files with clinicians:")
    print(f"   - {output_path}")
    print("   - data.csv (from export)")
    print("   - config.json (from export)")
    print("\nUsers can then open the HTML file and load their data.")

    return 0


if __name__ == "__main__":
    exit(main())
