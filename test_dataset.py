from dataset import load_dataset

train_loader, val_loader, classes = load_dataset()

print("Number of Classes :", len(classes))

print("\nClasses:\n")

for c in classes:
    print(c)

print("\nTraining Images :", len(train_loader.dataset))
print("Validation Images :", len(val_loader.dataset))