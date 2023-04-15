import { Dataset } from "./dataset";
import iris_dataset from "./data/iris.json";
import breast_cancer_dataset from "./data/breast_cancer.json";
import dry_beans_dataset from "./data/dry_beans.json";
import { KNN, VRKNN } from "./lib";

// Iris Dataset

console.log("Iris Dataset Predictions");

interface IrisDataPoint {
  sepalLength: number;
  sepalWidth: number;
  petalLength: number;
  petalWidth: number;
  species: string;
}

const iris_data: IrisDataPoint[] = iris_dataset;

const [training_data, testing_data, testing_labels] = Dataset.train_test_split({
  data: iris_data,
  training_entities: ["sepalLength", "sepalWidth"],
  testing_entity: "species",
  test_size: 0.2,
  random_state: 42,
});

const knn = new KNN({
  k: 3,
});

knn.fit({
  training_data: training_data,
});

const predictions = knn.predict({
  testing_data: testing_data,
});

const accuracy_knn = knn.evaluate({
  predictions: predictions,
  testing_labels: testing_labels,
});

const vrknn = new VRKNN({
  k: 3,
  radius_delta: 0.1,
});

vrknn.fit({
  training_data: training_data,
});

const vr_predictions = vrknn.predict({
  testing_data: testing_data,
});

const accuracy_vrknn = vrknn.evaluate({
  predictions: vr_predictions,
  testing_labels: testing_labels,
});

console.log("Total Iris Dataset: ", iris_data.length);
console.log("Length of training data of Iris Dataset: ", training_data.length);
console.log("Length of testing data of Iris Dataset: ", testing_data.length);
console.log("Accuracy of KNN: ", accuracy_knn);
console.log("Time taken to predict KNN: ", knn.time_taken_to_predict, "ms");
console.log("Accuracy of VRKNN: ", accuracy_vrknn);
console.log("Time taken to predict VRKNN: ", vrknn.time_taken_to_predict, "ms");

// Breast Cancer Dataset
interface BreastCancerDataPoint {
  radius_mean: number;
  texture_mean: number;
  diagnosis: string;
}

const breast_cancer_data: BreastCancerDataPoint[] = breast_cancer_dataset;

const [
  training_data_breast_cancer,
  testing_data_breast_cancer,
  testing_labels_breast_cancer,
] = Dataset.train_test_split({
  data: breast_cancer_data,
  training_entities: ["radius_mean", "texture_mean"],
  testing_entity: "diagnosis",
  test_size: 0.2,
  random_state: 42,
});

const knn_breast_cancer = new KNN({
  k: 3,
});

knn_breast_cancer.fit({
  training_data: training_data_breast_cancer,
});

const predictions_breast_cancer = knn_breast_cancer.predict({
  testing_data: testing_data_breast_cancer,
});

const accuracy_knn_breast_cancer = knn_breast_cancer.evaluate({
  predictions: predictions_breast_cancer,
  testing_labels: testing_labels_breast_cancer,
});

const vrknn_breast_cancer = new VRKNN({
  k: 3,
  radius_delta: 0.1,
});

vrknn_breast_cancer.fit({
  training_data: training_data_breast_cancer,
});

const vr_predictions_breast_cancer = vrknn_breast_cancer.predict({
  testing_data: testing_data_breast_cancer,
});

const accuracy_vrknn_breast_cancer = vrknn_breast_cancer.evaluate({
  predictions: vr_predictions_breast_cancer,
  testing_labels: testing_labels_breast_cancer,
});
console.log("\nBreast Cancer Wisconsin (Diagnostic) Data Set Predictions");
console.log("Total Breast Cancer Dataset: ", breast_cancer_data.length);
console.log(
  "Length of training data of Breast Cancer Dataset: ",
  training_data_breast_cancer.length
);
console.log(
  "Length of testing data of Breast Cancer Dataset:",
  testing_data_breast_cancer.length
);
console.log("Accuracy of KNN: ", accuracy_knn_breast_cancer);
console.log(
  "Time taken to predict KNN: ",
  knn_breast_cancer.time_taken_to_predict,
  "ms"
);
console.log("Accuracy of VRKNN: ", accuracy_vrknn_breast_cancer);
console.log(
  "Time taken to predict VRKNN: ",
  vrknn_breast_cancer.time_taken_to_predict,
  "ms"
);

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
  test_size: 0.2,
  random_state: 42,
});

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
  radius_delta: 20,
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
console.log("Accuracy of KNN: ", accuracy_knn_dry_beans);
console.log(
  "Time taken to predict KNN: ",
  knn_dry_beans.time_taken_to_predict,
  "ms"
);
console.log("Accuracy of VRKNN: ", accuracy_vrknn_dry_beans);
console.log(
  "Time taken to predict VRKNN: ",
  vrknn_dry_beans.time_taken_to_predict,
  "ms"
);
