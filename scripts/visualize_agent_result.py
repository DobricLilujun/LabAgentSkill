"""
Visualize Agent Result as HTML
Generates an interactive HTML visualization of LangChain/LangGraph agent results
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


def visualize_agent_result(result: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Generate an HTML visualization of agent result
    
    Args:
        result: The agent.invoke() result dictionary
        output_path: Optional path to save HTML file. If None, returns HTML string
    
    Returns:
        HTML string or file path if saved
    """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Result Visualization</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }}
        
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 14px;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section {{
            margin-bottom: 30px;
        }}
        
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        
        .section-count {{
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .message {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }}
        
        .message.user {{
            border-left-color: #28a745;
            background: #f0f8f5;
        }}
        
        .message.assistant {{
            border-left-color: #667eea;
            background: #f0f3ff;
        }}
        
        .message.system {{
            border-left-color: #ffc107;
            background: #fffbf0;
        }}
        
        .message-header {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            font-weight: 600;
            font-size: 13px;
        }}
        
        .role-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 11px;
            font-weight: 600;
            margin-right: 10px;
            text-transform: uppercase;
        }}
        
        .role-badge.user {{
            background: #28a745;
            color: white;
        }}
        
        .role-badge.assistant {{
            background: #667eea;
            color: white;
        }}
        
        .role-badge.system {{
            background: #ffc107;
            color: #333;
        }}
        
        .message-content {{
            color: #333;
            line-height: 1.6;
            word-break: break-word;
        }}
        
        .code-block {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.5;
            margin-top: 10px;
        }}
        
        .json-block {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.5;
        }}
        
        .collapsible {{
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px;
            background: #f0f3ff;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 3px solid #667eea;
        }}
        
        .collapsible:hover {{
            background: #e8ebff;
        }}
        
        .collapsible.active {{
            background: #e8ebff;
        }}
        
        .collapse-icon {{
            display: inline-block;
            margin-right: 8px;
            font-size: 12px;
        }}
        
        .collapse-icon.open {{
            transform: rotate(90deg);
        }}
        
        .collapsed-content {{
            display: none;
            margin-top: 10px;
        }}
        
        .collapsed-content.open {{
            display: block;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: 600;
            color: #333;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 15px 30px;
            border-top: 1px solid #dee2e6;
            font-size: 12px;
            color: #666;
        }}
        
        @media (max-width: 768px) {{
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 20px;
            }}
            
            .content {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Agent Result Visualization</h1>
            <p>Interactive view of LangChain/LangGraph agent execution</p>
        </div>
        
        <div class="content">
            {_generate_stats_html(result)}
            {_generate_messages_html(result)}
            {_generate_raw_data_html(result)}
        </div>
        
        <div class="footer">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            <a href="#" onclick="window.location.reload(); return false;">Refresh</a>
        </div>
    </div>
    
    <script>
        document.querySelectorAll('.collapsible').forEach(el => {{
            el.addEventListener('click', function() {{
                this.classList.toggle('active');
                this.nextElementSibling.classList.toggle('open');
                this.querySelector('.collapse-icon').classList.toggle('open');
            }});
        }});
    </script>
</body>
</html>"""
    
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html_content, encoding='utf-8')
        print(f"✓ Visualization saved to: {path.absolute()}")
        return str(path.absolute())
    else:
        return html_content


def _generate_stats_html(result: Dict[str, Any]) -> str:
    """Generate statistics section"""
    msg_count = len(result.get('messages', []))
    
    html = '<div class="section"><div class="stats">'
    html += f'<div class="stat-card"><div class="stat-label">Total Messages</div><div class="stat-value">{msg_count}</div></div>'
    
    user_msgs = len([m for m in result.get('messages', []) if m.get('role') == 'user'])
    assistant_msgs = len([m for m in result.get('messages', []) if m.get('role') == 'assistant'])
    
    html += f'<div class="stat-card"><div class="stat-label">User Messages</div><div class="stat-value">{user_msgs}</div></div>'
    html += f'<div class="stat-card"><div class="stat-label">Assistant Messages</div><div class="stat-value">{assistant_msgs}</div></div>'
    
    html += '</div></div>'
    return html


def _generate_messages_html(result: Dict[str, Any]) -> str:
    """Generate messages section"""
    messages = result.get('messages', [])
    
    if not messages:
        return '<div class="section"><div class="section-title">Messages</div><p>No messages found</p></div>'
    
    html = f'<div class="section"><div class="section-title">Messages <span class="section-count">{len(messages)}</span></div>'
    
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        # Handle different content types
        if isinstance(content, dict):
            content_str = json.dumps(content, indent=2)
            content_html = f'<div class="json-block"><pre>{_escape_html(content_str)}</pre></div>'
        elif isinstance(content, list):
            content_str = json.dumps(content, indent=2)
            content_html = f'<div class="json-block"><pre>{_escape_html(content_str)}</pre></div>'
        else:
            content_html = f'<div class="message-content">{_escape_html(str(content))}</div>'
        
        html += f'''
        <div class="message {role}">
            <div class="message-header">
                <span class="role-badge {role}">{role}</span>
                <span style="color: #999; font-size: 12px;">Message {i}</span>
            </div>
            {content_html}
        </div>
        '''
    
    html += '</div>'
    return html


def _generate_raw_data_html(result: Dict[str, Any]) -> str:
    """Generate raw data section with collapsible JSON"""
    raw_json = json.dumps(result, indent=2, default=str)
    
    html = f'''
    <div class="section">
        <div class="section-title">Raw Data</div>
        <div class="collapsible">
            <span><span class="collapse-icon">▶</span>Show Full Result JSON</span>
        </div>
        <div class="collapsed-content">
            <div class="json-block">
                <pre>{_escape_html(raw_json)}</pre>
            </div>
        </div>
    </div>
    '''
    return html


def _escape_html(text: str) -> str:
    """Escape HTML special characters"""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def save_result_to_html(result: Dict[str, Any], filename: str = "agent_result.html") -> str:
    """
    Convenience function to save result to HTML file
    
    Args:
        result: The agent result dictionary
        filename: Output filename (relative to current directory or absolute path)
    
    Returns:
        Path to saved file
    """
    return visualize_agent_result(result, output_path=filename)


# Example usage
if __name__ == "__main__":
    # Mock result for testing
    mock_result = {
        "messages": [
            {
                "role": "user",
                "content": "Write a SQL query to find all customers who made orders over $1000"
            },
            {
                "role": "assistant",
                "content": "SELECT * FROM customers WHERE order_amount > 1000"
            }
        ]
    }
    
    # Generate HTML
    html_path = save_result_to_html(mock_result, "test_result.html")
    print(f"Saved to: {html_path}")
