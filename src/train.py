
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data import ECGDataModule
from model import ECGClassifier

def main():
    pl.seed_everything(42)

    BATCH_SIZE = 32
    MAX_EPOCHS = 4
    LEARNING_RATE = 1e-4
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'time_series')

    data_module = ECGDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE)

    model = ECGClassifier(num_classes=4, learning_rate=LEARNING_RATE)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='ecg-resnet-{epoch:02d}-{val_loss:.2f}',
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
