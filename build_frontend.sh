set -ex

cd frontend
pnpm install
STATIC_EXPORT=1 pnpm next build
mkdir -p ../hibiri/static
cp -r out/* ../hibiri/static/