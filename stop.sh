#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ -f qwengine.pid ] && kill "$(cat qwengine.pid)" 2>/dev/null; then
  rm -f qwengine.pid; echo "stopped"
else
  rm -f qwengine.pid; echo "not running"
fi
