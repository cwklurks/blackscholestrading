"""UI components and styling for the Black-Scholes Trader app."""

# Custom CSS for Streamlit
CUSTOM_CSS = """
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Subtle card styling */
    .metric-card {
        background-color: #252526;
        border: 1px solid #3e3e42;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .metric-card h3 {
        margin-top: 0;
        font-size: 1rem;
        color: #aaaaaa;
        font-weight: 500;
    }
    .metric-card h2 {
        margin: 5px 0 0 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Clean up dataframe styling */
    .stDataFrame {
        border: 1px solid #3e3e42;
        border-radius: 4px;
    }
</style>
"""

# Lucide Icons (v0.344.0)
ICONS = {
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>',
    "file-text": '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><line x1="10" y1="9" x2="8" y2="9"></line>',
    "calculator": '<rect width="16" height="20" x="4" y="2" rx="2"></rect><line x1="8" y1="6" x2="16" y2="6"></line><line x1="16" y1="14" x2="16" y2="14"></line><line x1="16" y1="18" x2="16" y2="18"></line><line x1="12" y1="14" x2="12" y2="14"></line><line x1="12" y1="18" x2="12" y2="18"></line><line x1="8" y1="14" x2="8" y2="14"></line><line x1="8" y1="18" x2="8" y2="18"></line>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>',
    "wifi": '<path d="M5 12.55a11 11 0 0 1 14.08 0"></path><path d="M1.42 9a16 16 0 0 1 21.16 0"></path><path d="M8.53 16.11a6 6 0 0 1 6.95 0"></path><line x1="12" y1="20" x2="12.01" y2="20"></line>'
}


def get_icon(name: str, size: int = 20, color: str = "currentColor", mode: str = "html") -> str:
    """Generate an SVG icon from the Lucide icon set.
    
    Args:
        name: Icon name (e.g., 'activity', 'calculator')
        size: Icon size in pixels
        color: Icon color (CSS color value)
        mode: 'html' for inline SVG, 'url' for markdown image URL
        
    Returns:
        SVG markup string or markdown image URL
    """
    if mode == "url":
        url_color = "white" if color == "currentColor" else color.replace("#", "%23")
        return f"![{name}](https://api.iconify.design/lucide/{name}.svg?color={url_color}&height={size})"

    svg_content = ICONS.get(name, "")
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;">{svg_content}</svg>'


def get_chart_layout(title: str = "", height: int = 400) -> dict:
    """Get a consistent Plotly chart layout configuration.
    
    Args:
        title: Chart title
        height: Chart height in pixels
        
    Returns:
        Dictionary of Plotly layout options
    """
    return dict(
        title=title,
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="sans-serif", size=12, color="#e0e0e0")
    )

