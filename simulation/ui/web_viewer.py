"""
Web-based visualization viewer.

Perfect for remote development, Docker, or headless servers:
- No X11 needed
- View in browser (localhost:8080)
- Works with SSH tunneling
- Great for cloud/Docker deployments
"""

import numpy as np
import cv2
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import time


class WebViewer:
    """
    Web-based image viewer using HTTP streaming.

    View at: http://localhost:8080
    Perfect for remote development, Docker, or headless servers.
    """

    def __init__(self, port: int = 8080, quiet: bool = False):
        """
        Initialize web viewer.

        Args:
            port: HTTP server port
            quiet: If True, suppress URL output (prevents auto-open in some terminals)
        """
        self.port = port
        self.quiet = quiet
        self.latest_image: np.ndarray | None = None
        self.server: HTTPServer | None = None
        self.server_thread: threading.Thread | None = None
        self.running = False

        # Statistics
        self.frame_count = 0
        self.last_update = time.time()

        # Action callbacks
        self.action_callbacks = {}
        self.paused = False

    def register_action(self, action_name: str, callback):
        """
        Register an action callback.

        Args:
            action_name: Name of the action (e.g., 'respawn', 'pause')
            callback: Function to call when action is triggered
        """
        self.action_callbacks[action_name] = callback

    def start(self):
        """Start the web server."""
        # Create request handler with access to viewer
        viewer_self = self

        class ViewerRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Enable HTTP logs for debugging
                print(f"[HTTP] {format % args}")

            def do_POST(self):
                """Handle POST requests for actions."""
                print(f"\n[WebViewer] ===== POST REQUEST RECEIVED =====")
                print(f"[WebViewer] Path: {self.path}")
                if self.path == '/action':
                    # Read request body
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)

                    try:
                        import json
                        data = json.loads(post_data.decode('utf-8'))
                        action = data.get('action')
                        print(f"[WebViewer] Action requested: '{action}'")
                        print(f"[WebViewer] Registered actions: {list(viewer_self.action_callbacks.keys())}")

                        if action in viewer_self.action_callbacks:
                            print(f"[WebViewer] Executing action: '{action}'")
                            # Execute callback
                            if viewer_self.action_callbacks[action]():
                                print(f"[WebViewer] Action '{action}' completed")

                                # Send success response
                                self.send_response(200)
                                self.send_header('Content-Type', 'application/json')
                                self.end_headers()
                                response = json.dumps({'status': 'ok', 'action': action})
                                self.wfile.write(response.encode())
                            else:
                                print(f"[WebViewer] Action '{action}' has failed")
                                self.send_error(500)
                                self.end_headers()
                                response = json.dumps({'status': 'Internal Server Error', 'action': action})
                                self.wfile.write(response.encode())
                        else:
                            # Unknown action
                            print(f"[WebViewer] ERROR: Unknown action '{action}'")
                            self.send_response(400)
                            self.send_header('Content-Type', 'application/json')
                            self.end_headers()
                            response = json.dumps({'status': 'error', 'message': f'Unknown action: {action}'})
                            self.wfile.write(response.encode())
                    except Exception as e:
                        print(f"[WebViewer] ERROR processing action: {e}")
                        import traceback
                        traceback.print_exc()
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = json.dumps({'status': 'error', 'message': str(e)})
                        self.wfile.write(response.encode())
                else:
                    self.send_error(404)

            def do_GET(self):
                # Debug: Log first few requests
                if not hasattr(viewer_self, '_request_count'):
                    viewer_self._request_count = 0
                viewer_self._request_count += 1
                if viewer_self._request_count <= 10:
                    print(f"[WebViewer] Request #{viewer_self._request_count}: {self.path}")
                if self.path == '/':
                    # Serve HTML page
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    html = self._get_html()
                    self.wfile.write(html.encode())

                elif self.path == '/stream':
                    # Serve MJPEG stream
                    if viewer_self._request_count <= 10:
                        print("[WebViewer] Stream request received")
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.end_headers()

                    frame_sent = 0
                    try:
                        while viewer_self.running:
                            if viewer_self.latest_image is not None:
                                # Encode frame as JPEG
                                success, buffer = cv2.imencode('.jpg', viewer_self.latest_image,
                                                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                                if success:
                                    frame_bytes = buffer.tobytes()

                                    # Send frame with MJPEG multipart format
                                    self.wfile.write(b'--jpgboundary\r\n')
                                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                                    self.wfile.write(f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode())
                                    self.wfile.write(frame_bytes)
                                    self.wfile.write(b'\r\n')

                                    frame_sent += 1
                                    if frame_sent <= 3:
                                        print(f"[WebViewer] Sent frame {frame_sent} ({len(frame_bytes)} bytes)")

                            time.sleep(0.033)  # ~30 FPS
                    except Exception as e:
                        print(f"[WebViewer] Stream ended: {e}")

                elif self.path == '/favicon.ico':
                    # Return empty favicon to prevent browser from retrying
                    self.send_response(204)  # No Content
                    self.end_headers()

                else:
                    self.send_error(404)

            def _get_html(self):
                return f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Lane Detection Viewer</title>
                    <style>
                        body {{
                            margin: 0;
                            padding: 20px;
                            background: #1e1e1e;
                            color: #ffffff;
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                        }}
                        h1 {{
                            margin: 0 0 20px 0;
                            font-size: 24px;
                            font-weight: 300;
                        }}
                        .container {{
                            max-width: 1200px;
                            width: 100%;
                        }}
                        .video-container {{
                            position: relative;
                            width: 100%;
                            background: #000;
                            border-radius: 8px;
                            overflow: hidden;
                            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                        }}
                        img {{
                            width: 100%;
                            height: auto;
                            display: block;
                        }}
                        .controls {{
                            margin-top: 20px;
                            padding: 15px;
                            background: #2d2d2d;
                            border-radius: 8px;
                            display: flex;
                            gap: 10px;
                            flex-wrap: wrap;
                        }}
                        .btn {{
                            padding: 10px 20px;
                            background: #4a4a4a;
                            border: none;
                            border-radius: 4px;
                            color: #fff;
                            cursor: pointer;
                            font-size: 14px;
                            transition: background 0.2s;
                        }}
                        .btn:hover {{
                            background: #5a5a5a;
                        }}
                        .btn:active {{
                            background: #3a3a3a;
                        }}
                        .btn.primary {{
                            background: #2196F3;
                        }}
                        .btn.primary:hover {{
                            background: #1976D2;
                        }}
                        .btn.warning {{
                            background: #FF9800;
                        }}
                        .btn.warning:hover {{
                            background: #F57C00;
                        }}
                        .info {{
                            margin-top: 20px;
                            padding: 15px;
                            background: #2d2d2d;
                            border-radius: 8px;
                            font-size: 14px;
                        }}
                        .info-item {{
                            margin: 5px 0;
                        }}
                        .status {{
                            display: inline-block;
                            width: 8px;
                            height: 8px;
                            border-radius: 50%;
                            background: #4caf50;
                            margin-right: 8px;
                            animation: pulse 2s infinite;
                        }}
                        .status.paused {{
                            background: #FF9800;
                        }}
                        @keyframes pulse {{
                            0%, 100% {{ opacity: 1; }}
                            50% {{ opacity: 0.5; }}
                        }}
                        .key {{
                            color: #888;
                            margin-right: 10px;
                        }}
                        .kbd {{
                            background: #3a3a3a;
                            padding: 2px 6px;
                            border-radius: 3px;
                            font-family: monospace;
                            font-size: 12px;
                        }}
                        .notification {{
                            position: fixed;
                            top: 20px;
                            right: 20px;
                            background: #4CAF50;
                            color: white;
                            padding: 15px 20px;
                            border-radius: 4px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                            opacity: 0;
                            transition: opacity 0.3s;
                            z-index: 1000;
                        }}
                        .notification.show {{
                            opacity: 1;
                        }}
                        .notification.error {{
                            background: #f44336;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🚗 Lane Detection Viewer</h1>
                        <div class="video-container">
                            <img src="/stream" alt="Lane Detection Stream">
                        </div>
                        <div class="controls">
                            <button class="btn primary" onclick="sendAction('respawn')">🔄 Respawn Vehicle</button>
                            <button class="btn warning" id="pauseBtn" onclick="togglePause()">⏸ Pause</button>
                        </div>
                        <div class="info">
                            <div class="info-item">
                                <span class="status" id="statusDot"></span>
                                <span class="key">Status:</span>
                                <span id="statusText">Streaming</span>
                            </div>
                            <div class="info-item">
                                <span class="key">Port:</span>
                                <span>{viewer_self.port}</span>
                            </div>
                            <div class="info-item">
                                <span class="key">Keyboard Shortcuts:</span>
                                <span><kbd class="kbd">R</kbd> Respawn | <kbd class="kbd">Space</kbd> Pause/Resume | <kbd class="kbd">Q</kbd> Quit (in terminal)</span>
                            </div>
                        </div>
                    </div>

                    <div class="notification" id="notification"></div>

                    <script>
                        let isPaused = false;

                        function sendAction(action) {{
                            fetch('/action', {{
                                method: 'POST',
                                headers: {{
                                    'Content-Type': 'application/json',
                                }},
                                body: JSON.stringify({{ action: action }})
                            }})
                            .then(response => response.json())
                            .then(data => {{
                                if (data.status === 'ok') {{
                                    showNotification('Action executed: ' + action);
                                }} else {{
                                    showNotification('Error: ' + data.message, true);
                                }}
                            }})
                            .catch(error => {{
                                showNotification('Network error: ' + error, true);
                            }});
                        }}

                        function togglePause() {{
                            isPaused = !isPaused;
                            sendAction(isPaused ? 'pause' : 'resume');

                            const btn = document.getElementById('pauseBtn');
                            const statusDot = document.getElementById('statusDot');
                            const statusText = document.getElementById('statusText');

                            if (isPaused) {{
                                btn.textContent = '▶️ Resume';
                                btn.classList.add('primary');
                                btn.classList.remove('warning');
                                statusDot.classList.add('paused');
                                statusText.textContent = 'Paused';
                            }} else {{
                                btn.textContent = '⏸ Pause';
                                btn.classList.remove('primary');
                                btn.classList.add('warning');
                                statusDot.classList.remove('paused');
                                statusText.textContent = 'Streaming';
                            }}
                        }}

                        function showNotification(message, isError = false) {{
                            const notif = document.getElementById('notification');
                            notif.textContent = message;
                            notif.className = 'notification show' + (isError ? ' error' : '');

                            setTimeout(() => {{
                                notif.classList.remove('show');
                            }}, 3000);
                        }}

                        // Keyboard shortcuts
                        document.addEventListener('keydown', function(event) {{
                            // Prevent default for our shortcuts
                            if (event.key === 'r' || event.key === 'R' || event.key === ' ') {{
                                event.preventDefault();
                            }}

                            if (event.key === 'r' || event.key === 'R') {{
                                sendAction('respawn');
                            }} else if (event.key === ' ') {{
                                togglePause();
                            }}
                        }});

                        // Focus on body to ensure keyboard events are captured
                        document.body.focus();
                    </script>
                </body>
                </html>
                """

        # Start server in separate thread
        # Use ThreadingHTTPServer to handle multiple connections (stream + POST requests)
        self.server = ThreadingHTTPServer(('', self.port), ViewerRequestHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.running = True
        self.server_thread.start()

        if not self.quiet:
            print(f"\n✓ Web viewer started")
            # Format URL in a way that's less likely to auto-open (VSCode/terminal detection)
            print(f"  View at: http://localhost:{self.port}")
            print(f"  (Copy and paste the URL manually into your browser)")
            print(f"  Press Ctrl+C to stop\n")
        else:
            print(f"✓ Web viewer started on port {self.port}")

    def update(self, image: np.ndarray):
        """
        Update the displayed image.

        Args:
            image: RGB image array (H, W, 3)
        """
        # Convert RGB to BGR for OpenCV encoding
        if image is not None and len(image.shape) == 3:
            self.latest_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.frame_count += 1

            # Debug: print first few frames
            if self.frame_count <= 5:
                print(f"[WebViewer] Received frame {self.frame_count}, shape: {image.shape}, dtype: {image.dtype}, "
                      f"min: {image.min()}, max: {image.max()}")
        else:
            print(f"[WebViewer] WARNING: Invalid image - shape: {image.shape if image is not None else 'None'}")

    def show(self, image: np.ndarray) -> bool:
        """
        Show image (wrapper for update() to match viewer interface).

        Args:
            image: RGB image array (H, W, 3)

        Returns:
            True if should continue, False if quit requested
        """
        if self.frame_count == 0:
            print(f"[WebViewer] show() called for first time, image shape: {image.shape if image is not None else 'None'}")
        self.update(image)
        return self.is_running()

    def stop(self):
        """Stop the web server."""
        self.running = False
        if self.server:
            self.server.shutdown()
        print("Web viewer stopped")

    def close(self):
        """Close viewer (wrapper for stop() to match viewer interface)."""
        self.stop()

    def is_running(self) -> bool:
        """Check if viewer is running."""
        return self.running


# Simple test
if __name__ == "__main__":
    import time

    viewer = WebViewer(port=8080)
    viewer.start()

    print("Generating test video stream...")
    print("View at: http://localhost:8080")
    print("Press Ctrl+C to stop")

    try:
        frame_num = 0
        while True:
            # Create animated test image
            test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

            # Add frame number
            cv2.putText(test_image, f"Frame {frame_num}",
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            viewer.update(test_image)
            time.sleep(0.03)  # ~30 FPS
            frame_num += 1

    except KeyboardInterrupt:
        print("\nStopping...")
        viewer.stop()
