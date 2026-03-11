"""Development server. For production (multi-worker), use Gunicorn from this directory:
    gunicorn -w 4 --bind 0.0.0.0:5000 wsgi:app
"""
from app import create_app
from app.config import DefaultConfig


def main() -> None:
    app = create_app(DefaultConfig)
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()

