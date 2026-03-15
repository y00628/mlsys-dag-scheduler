CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -I third_party
SRCDIR   := source
SOURCES  := $(SRCDIR)/main.cpp $(SRCDIR)/problem.cpp $(SRCDIR)/solution.cpp \
            $(SRCDIR)/evaluator.cpp $(SRCDIR)/solver.cpp

# Default: debug build
.PHONY: all clean release submission

all: mlsys

mlsys: $(SOURCES)
	$(CXX) $(CXXFLAGS) -g -O0 -o $@ $^

# Optimized + static linked (for submission)
release: $(SOURCES)
	$(CXX) $(CXXFLAGS) -O2 -DNDEBUG -static -o mlsys $^

# Package for submission
TEAM_NAME ?= MyTeam
SUBMISSION_NUM ?= 1
submission: release
	@echo "Packaging submission..."
	@mkdir -p /tmp/submission_pkg/source
	@cp mlsys /tmp/submission_pkg/
	@cp $(SRCDIR)/*.cpp $(SRCDIR)/*.h /tmp/submission_pkg/source/
	@cp Makefile /tmp/submission_pkg/source/
	@cp -r third_party /tmp/submission_pkg/source/
	@echo "TODO: add writeup.pdf to /tmp/submission_pkg/"
	@cd /tmp/submission_pkg && zip -r $(CURDIR)/$(TEAM_NAME)_TrackA_$(SUBMISSION_NUM).zip .
	@rm -rf /tmp/submission_pkg
	@echo "Created: $(TEAM_NAME)_TrackA_$(SUBMISSION_NUM).zip"

clean:
	rm -f mlsys *.zip
