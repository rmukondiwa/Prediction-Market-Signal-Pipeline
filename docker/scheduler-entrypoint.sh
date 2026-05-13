#!/bin/sh
set -e
crontab /app/docker/crontab
exec cron -f
