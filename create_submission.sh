#!/bin/bash
if test -f "reddit_answer.txt"; then
    cp reddit_answer.txt answer.txt
    zip "reddit_answer.zip" answer.txt
fi
if test -f "twitter_answer.txt"; then
    cp twitter_answer.txt answer.txt
    zip "twitter_answer.zip" answer.txt
fi
rm answer.txt