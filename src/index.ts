import iris_dataset from "./data/iris.json";
import breast_cancer_dataset from "./data/breast_cancer.json";
import dry_beans_dataset from "./data/dry_beans.json";
import magic_gamma_dataset from "./data/magic_gamma.json";
import { KNN, VRKNN, Dataset } from "./lib";

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
const iris_k_value = 3;
const radius_delta = 0.1;

const knn = new KNN({
  k: iris_k_value,
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
  k: iris_k_value,
  radius_delta: radius_delta,
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
console.log("K value of Iris Dataset: ", iris_k_value);
console.log("Radius delta of Iris Dataset: ", radius_delta);
console.log("Accuracy of KNN: ", accuracy_knn);
console.log(
  "Time taken to predict using KNN: ",
  knn.time_taken_to_predict,
  "ms"
);
console.log("Accuracy of VRKNN: ", accuracy_vrknn);
console.log(
  "Time taken to predict using VRKNN: ",
  vrknn.time_taken_to_predict,
  "ms"
);

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

// MAGIC Gamma Telescope Dataset
interface MagicGammaTelescopeDataPoint {
  fLength: number;
  fWidth: number;
  class: string;
}

const magic_gamma_telescope_data: MagicGammaTelescopeDataPoint[] =
  magic_gamma_dataset as MagicGammaTelescopeDataPoint[];

const [
  training_data_magic_gamma_telescope,
  testing_data_magic_gamma_telescope,
  testing_labels_magic_gamma_telescope,
] = Dataset.train_test_split({
  data: magic_gamma_telescope_data,
  training_entities: ["fLength", "fWidth"],
  testing_entity: "class",
  test_size: 0.1,
  random_state: 42,
});

const magic_gamma_telescope_k_value = 10;
const magic_gamma_telescope_radius_delta = 0.1;

const knn_magic_gamma_telescope = new KNN({
  k: magic_gamma_telescope_k_value,
});

knn_magic_gamma_telescope.fit({
  training_data: training_data_magic_gamma_telescope,
});

const predictions_magic_gamma_telescope = knn_magic_gamma_telescope.predict({
  testing_data: testing_data_magic_gamma_telescope,
});

const accuracy_knn_magic_gamma_telescope = knn_magic_gamma_telescope.evaluate({
  predictions: predictions_magic_gamma_telescope,
  testing_labels: testing_labels_magic_gamma_telescope,
});

const vrknn_magic_gamma_telescope = new VRKNN({
  k: magic_gamma_telescope_k_value,
  radius_delta: magic_gamma_telescope_radius_delta,
});

vrknn_magic_gamma_telescope.fit({
  training_data: training_data_magic_gamma_telescope,
});

const vr_predictions_magic_gamma_telescope =
  vrknn_magic_gamma_telescope.predict({
    testing_data: testing_data_magic_gamma_telescope,
  });

const accuracy_vrknn_magic_gamma_telescope =
  vrknn_magic_gamma_telescope.evaluate({
    predictions: vr_predictions_magic_gamma_telescope,
    testing_labels: testing_labels_magic_gamma_telescope,
  });

console.log("\nMAGIC Gamma Telescope Dataset Predictions");
console.log(
  "Total MAGIC Gamma Telescope Dataset: ",
  magic_gamma_telescope_data.length
);
console.log(
  "Length of training data of MAGIC Gamma Telescope Dataset: ",
  training_data_magic_gamma_telescope.length
);

console.log(
  "Length of testing data of MAGIC Gamma Telescope Dataset: ",
  testing_data_magic_gamma_telescope.length
);

console.log(
  "K value of MAGIC Gamma Telescope Dataset: ",
  magic_gamma_telescope_k_value
);

console.log(
  "Radius delta of MAGIC Gamma Telescope Dataset: ",
  magic_gamma_telescope_radius_delta
);

console.log("Accuracy of KNN: ", accuracy_knn_magic_gamma_telescope);
console.log(
  "Time taken to predict using KNN: ",
  knn_magic_gamma_telescope.time_taken_to_predict,
  "ms"
);

console.log("Accuracy of VRKNN: ", accuracy_vrknn_magic_gamma_telescope);

console.log(
  "Time taken to predict using VRKNN: ",
  vrknn_magic_gamma_telescope.time_taken_to_predict,
  "ms"
);
