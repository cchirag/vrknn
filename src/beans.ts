import dry_beans_dataset from "./data/dry_beans.json";
import { KNN, VRKNN, Dataset } from "./lib";

// Dry Beans Dataset
interface DryBeansDataPoint {
  Area: number;
  Perimeter: number;
  Class: string;
}

const dry_beans_data: DryBeansDataPoint[] =
  dry_beans_dataset as DryBeansDataPoint[];

const [
  training_data_dry_beans,
  testing_data_dry_beans,
  testing_labels_dry_beans,
] = Dataset.train_test_split({
  data: dry_beans_data,
  training_entities: ["Area", "Perimeter"],
  testing_entity: "Class",
  test_size: 0.1,
  random_state: 42,
});

const dry_beans_k_value = 10;
const dry_beans_radius_delta = 0.1;

const knn_dry_beans = new KNN({
  k: 10,
});

knn_dry_beans.fit({
  training_data: training_data_dry_beans,
});

const predictions_dry_beans = knn_dry_beans.predict({
  testing_data: testing_data_dry_beans,
});

const accuracy_knn_dry_beans = knn_dry_beans.evaluate({
  predictions: predictions_dry_beans,
  testing_labels: testing_labels_dry_beans,
});

const vrknn_dry_beans = new VRKNN({
  k: 10,
  radius_delta: 0.1,
});

vrknn_dry_beans.fit({
  training_data: training_data_dry_beans,
});

const vr_predictions_dry_beans = vrknn_dry_beans.predict({
  testing_data: testing_data_dry_beans,
});

const accuracy_vrknn_dry_beans = vrknn_dry_beans.evaluate({
  predictions: vr_predictions_dry_beans,
  testing_labels: testing_labels_dry_beans,
});

console.log("\nDry Beans Dataset Predictions");
console.log("Total Dry Beans Dataset: ", dry_beans_data.length);
console.log(
  "Length of training data of Dry Beans Dataset: ",
  training_data_dry_beans.length
);
console.log(
  "Length of testing data of Dry Beans Dataset: ",
  testing_data_dry_beans.length
);
console.log("K value of Dry Beans Dataset: ", dry_beans_k_value);
console.log("Radius delta of Dry Beans Dataset: ", dry_beans_radius_delta);
console.log("Accuracy of KNN: ", accuracy_knn_dry_beans);
console.log(
  "Time taken to predict using KNN: ",
  knn_dry_beans.time_taken_to_predict,
  "ms"
);
console.log("Accuracy of VRKNN: ", accuracy_vrknn_dry_beans);
console.log(
  "Time taken to predict using VRKNN: ",
  vrknn_dry_beans.time_taken_to_predict,
  "ms"
);
