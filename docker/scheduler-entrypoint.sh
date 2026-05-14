#!/bin/sh
set -e
# Persist runtime env vars for cron jobs (cron doesn't inherit Docker env)
printenv | sed "s/'/'\\\\''/g; s/=\(.*\)/='\1'/" | sed 's/^/export /' > /app/.cron-env
crontab /app/docker/crontab
exec cron -f
