ifeq ($(PLATFORM),windows)
    CXX = g++
    EXE_EXT = .exe
    BOOST_INCLUDE = C:/local/boost_1_82_0
    RM = del /Q
    BIN_PREFIX =
else
    CXX = clang++
    EXE_EXT =
    BOOST_INCLUDE = /opt/homebrew/include
    RM = rm -f
    BIN_PREFIX = ./
endif

# --------------------------
# Select which source to build
# Default is rangetree; override with: make FILE=bfssp
# --------------------------
FILE ?= rangetree

# MODIFIED: Removed the "code/" prefix to match the flat directory structure
SRC = $(FILE).cpp
BIN = $(FILE)$(EXE_EXT)

CXXFLAGS = -std=c++17 -O2 -Wall -I"$(BOOST_INCLUDE)"

# Generic args (only used by 'run' and bfssp_run)
ARGS ?=

# --------------------------
# Build
# --------------------------
.PHONY: all
all: $(BIN)

$(BIN): $(SRC)
	@echo [BUILD] Compiling $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

# --------------------------
# Generic run (honors ARGS)
# --------------------------
.PHONY: run
run: all
	@echo [RUN] $(BIN_PREFIX)$(BIN) $(ARGS)
	$(BIN_PREFIX)$(BIN) $(ARGS)

# --------------------------
# No-args run helper
# --------------------------
.PHONY: run_noargs
run_noargs: all
	@echo [RUN] $(BIN_PREFIX)$(BIN)
	$(BIN_PREFIX)$(BIN)

# --------------------------
# Clean
# --------------------------
.PHONY: clean
clean:
	@echo [CLEAN] Removing built binaries
ifeq ($(PLATFORM),windows)
# MODIFIED: Target specific executables in the root directory
	$(RM) *.exe
else
# MODIFIED: Strictly target Unix binary outputs to prevent deleting source files
	$(RM) *.out *.o $(FILE) Ratio_Tversky Difference_Tversky bfssp
endif

# --------------------------
# Shortcuts
# --------------------------
# bfssp expects arguments
BFSSP_ARGS ?= dataset/sorted_texas_tribune.csv 23554.1 23554.3 2 0.00001
.PHONY: bfssp_run
bfssp_run:
	$(MAKE) FILE=bfssp PLATFORM=$(PLATFORM) run ARGS="$(BFSSP_ARGS)"

# rangetree has NO arguments
.PHONY: rangetree_Adult_run
rangetree_Adult_run:
	$(MAKE) FILE=Ratio_Tversky_Adult PLATFORM=$(PLATFORM) run_noargs

.PHONY: rangetree_Urban_run
rangetree_Urban_run:
	$(MAKE) FILE=Ratio_Tversky_Urban PLATFORM=$(PLATFORM) run_noargs

.PHONY: rangetree_Texas_run
rangetree_Texas_run:
	$(MAKE) FILE=Ratio_Tversky_Texas PLATFORM=$(PLATFORM) run_noargs

.PHONY: rangetree_Synthetic_run
rangetree_Synthetic_run:
	$(MAKE) FILE=Ratio_Tversky_Synthetic PLATFORM=$(PLATFORM) run_noargs

.PHONY: rangetree_Penguin_run
rangetree_Penguin_run:
	$(MAKE) FILE=Ratio_Tversky_Penguin PLATFORM=$(PLATFORM) run_noargs

# fairness has NO arguments
.PHONY: fairness_Adult_run
fairness_run:
	$(MAKE) FILE=Difference_Tversky_Binary_Adult PLATFORM=$(PLATFORM) run_noargs

.PHONY: fairness_Synthetic_run
fairness_synthetic_run:
	$(MAKE) FILE=Difference_Tversky_Binary_Synthetic PLATFORM=$(PLATFORM) run_noargs

.PHONY: fairness_Texas_run
fairness_Texas_run:
	$(MAKE) FILE=Difference_Tversky_Binary_Texas PLATFORM=$(PLATFORM) run_noargs

.PHONY: fairness_Penguin_run
fairness_Penguin_run:
	$(MAKE) FILE=Difference_Tversky_Binary_Penguin PLATFORM=$(PLATFORM) run_noargs

.PHONY: fairness_Urban_run
fairness_Urban_run:
	$(MAKE) FILE=Difference_Tversky_Binary_Urban PLATFORM=$(PLATFORM) run_noargs

PENGUIN_ARGS ?= sorted_penguins.csv 3
.PHONY: penguin_run
penguin_run:
	$(MAKE) FILE=Difference_Multicolor_Penguin PLATFORM=$(PLATFORM) run ARGS="$(PENGUIN_ARGS)"

URBAN_ARGS ?= UrbanDB.csv 3
.PHONY: urban_run
urban_run:
	$(MAKE) FILE=Difference_Multicolor_UrbanDB PLATFORM=$(PLATFORM) run ARGS="$(URBAN_ARGS)"

SYNTHETIC_ARGS ?= Synthetic.csv 3
.PHONY: synthetic_run
synthetic_run:
	$(MAKE) FILE=Difference_Multicolor_Synthetic PLATFORM=$(PLATFORM) run ARGS="$(SYNTHETIC_ARGS)"

#Difference Multicolor
#make penguin_run PLATFORM=windows PENGUIN_ARGS="../dataset/penguin/sorted_penguins.csv 3" - Good
#make urban_run PLATFORM=windows URBAN_ARGS="../dataset/UrbanDB.csv 3" - Good
#make synthetic_run PLATFORM=windows SYNTHETIC_ARGS="../dataset/synthetic_30k_unique.csv 3" - Good

#Ratio
#make rangetree_Adult_run PLATFORM=windows - Cancelled
#make rangetree_Urban_run PLATFORM=windows - Good
#make rangetree_Texas_run PLATFORM=windows - Need to check once again
#make rangetree_Synthetic_run PLATFORM=windows - Good
#make rangetree_Penguin_run PLATFORM=windows - Good

#Difference Binary Baseline
#make fairness_Adult_run PLATFORM=windows - Cancelled
#make fairness_synthetic_run PLATFORM=windows - Good
#make fairness_Texas_run PLATFORM=windows - Good
#make fairness_Urban_run PLATFORM=windows - Good



#make fairness_run PLATFORM=windows
#make synthetic_run PLATFORM=windows SYNTHETIC_ARGS="../dataset/synthetic_30k_unique.csv 3"