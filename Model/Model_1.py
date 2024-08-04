#!pip install transformers
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AdamW

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

# Random seed for reproducibilty
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading the dataset
dataset = pd.read_csv('/content/drive/MyDrive/Dataset-DL/fake reviews dataset.csv')

# create the class of Dataset to process data
class FakeReviewDataset(Dataset):
    def __init__(self,reviews,targets,tokenizer,max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self,item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation = True,
            pad_to_max_length=True,
            return_tensors='pt',
            )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)

# create the DataLoader to redesign the dataset into batch size
def create_data_loader(df,tokenizer,max_len,batch_size):
    ds = FakeReviewDataset(
        reviews = df.text_.to_numpy(),
        targets = df.label.to_numpy(),
        tokenizer = tokenizer,
        max_len=max_len)

    return DataLoader(ds,
                      batch_size=batch_size)


# Training function for each epoch on the training set
def train_epoch(model, data_loader, optimizer, device, scheduler, batch_size):
    model = model.train()
    losses = 0
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        # Get model outputs
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        # get the loss value
        loss = outputs.loss
        correct_predictions += torch.sum(preds == targets).item() / batch_size
        losses += loss

        # Backward prop
        loss.backward()

        # Gradient Descent
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    train_loss = (losses/len(data_loader)).item()
    train_acc = correct_predictions/len(data_loader)

    return train_acc , train_loss



# the evaluation function for the validation set
def eval_model(model, data_loader, device, batch_size):
    model = model.eval()

    losses = 0
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # Get model outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            loss = outputs.loss

            correct_predictions += torch.sum(preds == targets).item() / batch_size
            losses += loss

    val_loss = (losses/len(data_loader)).item()
    val_acc = correct_predictions/len(data_loader)

    return val_acc , val_loss

# the evaluation function for the test set
def test_set_predictions(model, data_loader):
    model = model.eval()

    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            # Get outouts
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            predictions.extend(preds)
            prediction_probs.extend(logits)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return predictions, prediction_probs, real_values

MAX_LEN = 150
BATCH_SIZE = 64
# Set the model name
MODEL_NAME = 'bert-base-uncased'

# Build a BERT based tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the basic BERT model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(device)

# split the dataset inot 3 parts
df_train, df_test = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

 # Create train, val, and test data loaders
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# Number of iterations
EPOCHS = 5

# Optimizer Adam
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)



history = {'train_loss': [],
           'train_acc': [],
           'val_loss': [],
           'val_acc': []}
best_accuracy = 0

for epoch in range(EPOCHS):

    # Show details
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        optimizer,
        device,
        scheduler,
        BATCH_SIZE
    )

    print(f"Train loss {train_loss} accuracy {train_acc}")

    # Get model performance (accuracy and loss)
    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        device,
        BATCH_SIZE
    )

    print(f"Val   loss {val_loss} accuracy {val_acc}")
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    # If we beat prev performance
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc
