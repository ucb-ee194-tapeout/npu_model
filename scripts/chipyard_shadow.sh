#!/usr/bin/env bash
# Make a shadow Chipyard tree that builds this atlas-npu checkout as generators/atlas-npu.
# The shadow tree has copies of the sbt build files and links to the other Chipyard
# directories. sbt writes all build outputs in the shadow tree, so the Chipyard checkout
# stays unchanged.
#
# Usage: scripts/chipyard_shadow.sh CHIPYARD SHADOW
#   CHIPYARD  a Chipyard checkout with initialized submodules (read only)
#   SHADOW    the directory to make
set -euo pipefail

[ $# -eq 2 ] || { echo "usage: $0 CHIPYARD SHADOW" >&2; exit 2; }
cy=$(realpath "$1")
rtl=$(realpath "$(dirname "$0")/../..")
mkdir -p "$2"
shadow=$(realpath "$2")

mkdir -p "$shadow"/{project,scripts,generators,.java_tmp}
cp "$cy"/build.sbt "$shadow"/
cp "$cy"/project/build.properties "$cy"/project/*.sbt "$shadow"/project/
cp "$cy"/scripts/sbt-launch.jar "$shadow"/scripts/
for dir in "$cy"/generators/*/; do
  name=$(basename "$dir")
  [ "$name" = atlas-npu ] || ln -sfn "${dir%/}" "$shadow/generators/$name"
done
ln -sfn "$rtl" "$shadow"/generators/atlas-npu
for dir in tools sims fpga; do
  ln -sfn "$cy/$dir" "$shadow/$dir"
done

# sbt writes target/ in the base directory of each project. Move them into the shadow tree.
cat > "$shadow"/project/ShadowTargets.scala <<'EOF'
import sbt._
import sbt.Keys._

object ShadowTargets extends AutoPlugin {
  override def trigger = allRequirements
  override def requires = plugins.JvmPlugin
  override def projectSettings = Seq(
    target := (ThisBuild / baseDirectory).value / "target" / "projects" / thisProject.value.id)
}
EOF
