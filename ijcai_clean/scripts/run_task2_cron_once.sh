#!/usr/bin/env bash
set -euo pipefail

CRON_MARKER="get_useful_task2_once_2026_05_05"

# Remove this one-shot cron entry before starting the long task.
tmp_cron="$(mktemp)"
crontab -l 2>/dev/null | grep -v "${CRON_MARKER}" > "${tmp_cron}" || true
crontab "${tmp_cron}"
rm -f "${tmp_cron}"

/home/wz/projects/mypro/get_useful/ijcai_clean/scripts/run_task2_cron.sh
