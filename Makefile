CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -I third_party
SRCDIR   := source
BUILDDIR := build
BINARY   := $(BUILDDIR)/mlsys
UNAME_S  := $(shell uname -s)
STATIC_LDFLAGS := -static

ifeq ($(UNAME_S),Darwin)
STATIC_LDFLAGS :=
endif
SOURCES  := $(SRCDIR)/main.cpp $(SRCDIR)/problem.cpp $(SRCDIR)/solution.cpp \
            $(SRCDIR)/evaluator.cpp $(SRCDIR)/solver.cpp \
            $(SRCDIR)/conv_accelerator.cpp $(SRCDIR)/optimus.cpp

# Default: debug build
.PHONY: all clean release submission

all: $(BINARY)

$(BINARY): $(SOURCES)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -g -O0 -o $@ $^

# Optimized + static linked (for submission)
release: $(SOURCES)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -O2 -DNDEBUG $(STATIC_LDFLAGS) -o $(BINARY) $^

# Package for submission
TEAM_NAME ?= MyTeam
SUBMISSION_NUM ?= 1
submission: release
	@echo "Packaging submission..."
	@mkdir -p /tmp/submission_pkg/source
	@cp $(BINARY) /tmp/submission_pkg/mlsys
	@cp $(SRCDIR)/*.cpp $(SRCDIR)/*.h /tmp/submission_pkg/source/
	@cp Makefile /tmp/submission_pkg/source/
	@cp -r third_party /tmp/submission_pkg/source/
	@echo "TODO: add writeup.pdf to /tmp/submission_pkg/"
	@cd /tmp/submission_pkg && zip -r $(CURDIR)/$(TEAM_NAME)_TrackA_$(SUBMISSION_NUM).zip .
	@rm -rf /tmp/submission_pkg
	@echo "Created: $(TEAM_NAME)_TrackA_$(SUBMISSION_NUM).zip"

clean:
	rm -rf $(BUILDDIR) *.zip
