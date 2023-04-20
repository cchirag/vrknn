import breast_cancer_dataset from "./data/breast_cancer.json";
import { KNN, VRKNN, Dataset } from "./lib";

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
  test_size: 0.1,
  random_state: 42,
});

const breast_cancer_k_value = 3;
const breast_cancer_radius_delta = 0.1;

const knn_breast_cancer = new KNN({
  k: breast_cancer_k_value,
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
  k: breast_cancer_k_value,
  radius_delta: breast_cancer_radius_delta,
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
console.log("K value of Breast Cancer Dataset: ", breast_cancer_k_value);
console.log(
  "Radius delta of Breast Cancer Dataset: ",
  breast_cancer_radius_delta
);
console.log("Accuracy of KNN: ", accuracy_knn_breast_cancer);
console.log(
  "Time taken to predict using KNN: ",
  knn_breast_cancer.time_taken_to_predict,
  "ms"
);
console.log("Accuracy of VRKNN: ", accuracy_vrknn_breast_cancer);
console.log(
  "Time taken to predict using VRKNN: ",
  vrknn_breast_cancer.time_taken_to_predict,
  "ms"
);
