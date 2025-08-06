# pdf_rename
#
# @file
# @version 0.1

# ANSI color codes
RED=\033[1;31m
GREEN=\033[1;32m
YELLOW=\033[1;33m
BLUE=\033[1;34m
CYAN=\033[1;36m
RESET=\033[0m

all: compile

compile: lint typecheck

lint:
	@echo "$(YELLOW)==> Running linter and formatter$(RESET)"
	@poetry run ruff format src
	@poetry run ruff check --fix src

typecheck:
	@echo "$(YELLOW)==> Running type checker (mypy)$(RESET)"
	@poetry run mypy src

# end
