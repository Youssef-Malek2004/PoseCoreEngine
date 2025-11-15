#!/usr/bin/env python3
"""
Simple HTTP server for Push-Up Counter web app.
Run this on your computer and access from your phone on the same network.
"""

import http.server
import socketserver
import socket
import os
import sys
import ssl


def get_local_ip():
    """Get the local IP address of this machine."""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "localhost"


def main():
    PORT = 8080

    # Change to the web directory
    web_dir = os.path.join(os.path.dirname(__file__), 'web')
    if os.path.exists(web_dir):
        os.chdir(web_dir)
    else:
        print(f"Error: web directory not found at {web_dir}")
        sys.exit(1)

    Handler = http.server.SimpleHTTPRequestHandler

    # Allow requests from any origin (needed for mobile access)
    class CORSRequestHandler(Handler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            super().end_headers()

    with socketserver.TCPServer(("0.0.0.0", PORT), CORSRequestHandler) as httpd:
        # Use the cert you just created (adjust file names)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        base_dir = os.path.dirname(__file__)
        cert_path = os.path.join(base_dir, "localhost+3.pem")
        key_path = os.path.join(base_dir, "localhost+3-key.pem")
        context.load_cert_chain(certfile=cert_path, keyfile=key_path)

        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

        local_ip = get_local_ip()
        print(f"\nOpen on phone: https://{local_ip}:{PORT}\n")  # note https
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            sys.exit(0)

        print("\n" + "=" * 60)
        print("🏋️  Push-Up Counter Web Server Running!")
        print("=" * 60)
        print(f"\n📱 Access from your Samsung S25 Ultra:")
        print(f"\n   Open browser and go to:")
        print(f"\n   http://{local_ip}:{PORT}")
        print(f"\n   (Make sure your phone is on the same WiFi network)")
        print("\n" + "=" * 60)
        print("\n💡 Tips:")
        print("   - Grant camera permissions when prompted")
        print("   - Use landscape mode for best experience")
        print("   - Use back camera for better quality")
        print("\n🛑 Press Ctrl+C to stop the server")
        print("=" * 60 + "\n")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()