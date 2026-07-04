#!/bin/bash

ps aux | grep "job command" | grep -v grep | sort -k2n
kill -15 PID