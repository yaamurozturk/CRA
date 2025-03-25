GITLAB_RAW_URL=https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv
LOCAL_FILE=retraction_watch.csv

FILE_HASH=$(shell [ -f $(LOCAL_FILE) ] && sha256sum $(LOCAL_FILE) | awk '{print $$1}' || echo "no_local_file")

check_update:
	@echo "Checking for updates..."
	@git_file_hash=$$(curl -s $(GITLAB_RAW_URL) | sha256sum | awk '{print $$1}'); \
	local_file_hash=$(FILE_HASH); \
	if [ "$$git_file_hash" != "$$local_file_hash" ]; then \
		echo "File changed! Downloading latest version..."; \
		curl -s -o $(LOCAL_FILE) $(GITLAB_RAW_URL); \
		echo "Updated $(LOCAL_FILE)"; \
	else \
		echo "Local file is already up to date."; \
	fi

update: check_update

.PHONY: check_update update

