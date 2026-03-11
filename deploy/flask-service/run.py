from app import create_app
from app.config import DefaultConfig


def main() -> None:
    app = create_app(DefaultConfig)
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()

