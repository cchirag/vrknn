import { TrainingDataPoint, TestDatapoint } from "models";
import { array_shuffle } from "./utils";

export class Dataset {
  public static train_test_split<T>(params: {
    data: T[];
    training_entities: Array<keyof T>;
    testing_entity: keyof T;
    test_size: 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9;
    random_state: number;
  }): [
    training_data: TrainingDataPoint[],
    testing_data: TestDatapoint[],
    testing_labels: string[]
  ] {
    const { data, training_entities, testing_entity, test_size, random_state } =
      params;
    const training_data: TrainingDataPoint[] = [];
    const testing_data: TestDatapoint[] = [];
    const testing_labels: string[] = [];
    const data_length = data.length;
    const test_data_length = Math.floor(data_length * test_size);
    const training_data_length = data_length - test_data_length;

    (() => {
      data.forEach((data_point, index) => {
        training_entities.forEach((entity) => {
          if (typeof data_point[entity] !== "number")
            throw new Error("All training entities must be numbers");
        });
        if (typeof data_point[testing_entity] !== "string")
          throw new Error("Testing entity must be a string");
      });
    })();

    // Shuffle the data
    const shuffled_data = array_shuffle(data, random_state);

    // Split the data into training and testing
    shuffled_data.forEach((data_point, index) => {
      if (index < training_data_length) {
        training_data.push({
          coordinates: training_entities.map(
            (entity) => data_point[entity] as number
          ),
          label: data_point[testing_entity] as string,
        });
      } else {
        testing_data.push({
          coordinates: training_entities.map(
            (entity) => data_point[entity] as number
          ),
        });
        testing_labels.push(data_point[testing_entity] as string);
      }
    });

    return [training_data, testing_data, testing_labels];
  }
}
