PROTO_DIR := src/huddle_drawings/proto
PROTOS := $(PROTO_DIR)/signaling_protocol.proto $(PROTO_DIR)/peer_message.proto

.PHONY: help proto test sync clean

help:
	@echo "proto  - regenerate the *_pb2.py modules from the vendored .proto files"
	@echo "test   - run the test suite"
	@echo "sync   - install dependencies"
	@echo "clean  - remove caches and generated protobuf modules"

proto:
	protoc --proto_path=$(PROTO_DIR) --python_out=$(PROTO_DIR) $(notdir $(PROTOS))
	@echo "regenerated: $(PROTO_DIR)/*_pb2.py"

test:
	uv run pytest -q

sync:
	uv sync

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f $(PROTO_DIR)/*_pb2.py
