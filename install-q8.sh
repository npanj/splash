#!/bin/sh
# Install the prebuilt Splash Q8 fork on an Apple Silicon Mac. No Xcode, no compiling.
#
#   curl -fsSL https://raw.githubusercontent.com/npanj/splash/q8/install-q8.sh | sh
#
# Downloads the release archive from GitHub, checks its SHA-256, unpacks it under
# ~/Library/Application Support/Splash-Q8/<version>, and writes a `splash-q8`
# command. It never touches an existing `splash` from Homebrew. Model weights live
# in the Hugging Face cache, shared with upstream Splash, so nothing is downloaded twice.
#
#   SPLASH_Q8_VERSION  release tag to install (default: the version below)
#   SPLASH_Q8_BASE_URL file server override, e.g. http://127.0.0.1:8123 for a local check
#   SPLASH_BIN_DIR     where to put `splash-q8` (default: Homebrew bin or ~/.local/bin)
set -eu

VERSION=${SPLASH_Q8_VERSION:-1.0-q8}
NAME="splash-$VERSION-arm64-macos26"
BASE=${SPLASH_Q8_BASE_URL:-https://github.com/npanj/splash/releases/download/$VERSION}
URL="$BASE/$NAME.tar.gz"
APP="$HOME/Library/Application Support/Splash-Q8"

fail() { echo "splash-q8 install: $*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || fail "needs an Apple Silicon Mac."
os=$(sw_vers -productVersion)
major=${os%%.*}
minor=${os#"$major"}
minor=${minor#.}
minor=${minor%%.*}
[ -n "$minor" ] || minor=0
[ "$major" -gt 26 ] 2>/dev/null || { [ "$major" -eq 26 ] && [ "$minor" -ge 4 ]; } \
    || fail "needs macOS 26.4 or newer; this Mac runs $os."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM
echo "Downloading $NAME.tar.gz ..."
curl -fL --progress-bar "$URL" -o "$tmp/$NAME.tar.gz" || fail "download failed: $URL"
curl -fsSL "$URL.sha256" -o "$tmp/sha256" || fail "checksum download failed"
want=$(tr -d '[:space:]' < "$tmp/sha256")
got=$(shasum -a 256 "$tmp/$NAME.tar.gz" | awk '{print $1}')
[ "$want" = "$got" ] || fail "checksum mismatch (expected $want, got $got)"

mkdir -p "$APP"
rm -rf "$APP/$VERSION.partial"
mkdir "$APP/$VERSION.partial"
tar -xzf "$tmp/$NAME.tar.gz" -C "$APP/$VERSION.partial" --strip-components 1
# A browser download would mark these files as quarantined; clear it so macOS runs them.
xattr -dr com.apple.quarantine "$APP/$VERSION.partial" 2>/dev/null || true
rm -rf "$APP/$VERSION"
mv "$APP/$VERSION.partial" "$APP/$VERSION"
ln -sfn "$VERSION" "$APP/current"

if [ -n "${SPLASH_BIN_DIR:-}" ]; then
    bin=$SPLASH_BIN_DIR
elif [ -d /opt/homebrew/bin ] && [ -w /opt/homebrew/bin ]; then
    bin=/opt/homebrew/bin
else
    bin="$HOME/.local/bin"
fi
mkdir -p "$bin"
cat > "$bin/splash-q8" <<EOF
#!/bin/sh
export PYTHONDONTWRITEBYTECODE=1
exec "$APP/current/python/bin/python3" -u "$APP/current/install/launcher.py" "\$@"
EOF
chmod 0755 "$bin/splash-q8"

"$bin/splash-q8" --version
echo
echo "Installed. Start the 8-bit model (first run downloads ~27 GB):"
echo "  splash-q8 serve --model nitinpanj/Qwen3.8-27B-Splash-HQ"
case ":$PATH:" in
    *":$bin:"*) ;;
    *) echo; echo "Note: $bin is not on your PATH. Add it, or run $bin/splash-q8 directly." ;;
esac
