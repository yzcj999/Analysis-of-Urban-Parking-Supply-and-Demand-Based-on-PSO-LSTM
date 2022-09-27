@echo off
::声明采用UTF-8编码
chcp 65001
title 【定时爬虫任务】
cd /d %~dp0
echo 当前路径: %cd%

python 新建文本文档.py

::清屏
cls
echo 当前路径: %cd%


::运行完批处理, 停留在cmd窗口
pause
::exit