# 既存のMakefileに以下を追加

.PHONY: clean-non-images
clean-non-images:
	@echo "画像ファイル以外を削除します..."
	@find data/complete/images -type f ! -name "*.jpg" ! -name "*.jpeg" ! -name "*.png" ! -name "*.gif" -ls
	@read -p "上記のファイルを削除してもよろしいですか？ (y/n) " yn; \
	case $$yn in \
		[Yy]* ) find data/complete/images -type f ! -name "*.jpg" ! -name "*.jpeg" ! -name "*.png" ! -name "*.gif" -delete;; \
		* ) echo "削除をキャンセルしました";; \
	esac

# 確認なしで直接削除する場合は以下のターゲットを使用
.PHONY: force-clean-non-images
force-clean-non-images:
	@echo "画像ファイル以外を削除します..."
	@find data/complete/images -type f ! -name "*.jpg" ! -name "*.jpeg" ! -name "*.png" ! -name "*.gif" -delete

.PHONY: train
train:
	@mkdir -p logs
	@echo "Starting training in background... (log: logs/train.log)"
	@nohup python exp/cli_250508_complete.py model-train 250508_complete_weightedCIoU_epoch10 > logs/250508_complete_weightedCIoU_epoch10.log 2>&1 &
	@echo "Training started. Check logs/train.log for progress."

.PHONY: preprocess
preprocess:
	@mkdir -p logs
	@echo "Starting training in background... (log: logs/train.log)"
	@nohup python exp/cli_preprocess.py preprocessing > logs/pre_complete.log 2>&1 &
	@echo "Training started. Check logs/train.log for progress."
