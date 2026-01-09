
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from data import EkgDataModule
from model import LitEfficientNetV2S

def main():
    pl.seed_everything(42)

    BATCH_SIZE = 32
    MAX_EPOCHS = 10
    LEARNING_RATE = 1e-4
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'processed')
    NUM_WORKERS = 4
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.2

    data_module = EkgDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        use_weighted_sampler=False,
    )

    model = LitEfficientNetV2S(num_classes=4, lr=LEARNING_RATE)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='ecg-efficientnetv2s-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto", 
        devices="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
    )

    print("Starting training...")
    trainer.fit(model, datamodule=data_module)
    
    print(f"Training completed. Best model path: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()