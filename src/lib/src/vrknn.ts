import { best_of_all, distance } from "../../utils";
import { TestDatapoint, TrainingDataPoint } from "../../models";
interface Bucket {
  id: number;
  data: TrainingDataPoint[];
}
class HashTable {
  private buckets: Bucket[] = [];
  private bucket_size: number = 0;

  public create_buckets = (params: { data: TrainingDataPoint[] }): void => {
    const { data } = params;
    const largest_x_coordinate = Math.max(
      ...data.map((data_point) => data_point.coordinates[0])
    );
    const smallest_x_coordinate = Math.min(
      ...data.map((data_point) => data_point.coordinates[0])
    );
    const range_of_x_coordinates = largest_x_coordinate - smallest_x_coordinate;
    const number_of_buckets = Math.ceil(1 + 3.322 * Math.log10(data.length));
    this.bucket_size = range_of_x_coordinates / number_of_buckets;
    for (let i = 0; i < number_of_buckets; i++) {
      const new_bucket: Bucket = {
        id: smallest_x_coordinate + i * this.bucket_size,
        data: [],
      };
      this.buckets.push(new_bucket);
    }
  };

  private hash_function(data_point: TrainingDataPoint): number {
    var bucket_id: number = 0;
    for (var i = 0; i < this.buckets.length; i++) {
      if (this.buckets[i].id + this.bucket_size < data_point.coordinates[0]) {
        if (i === this.buckets.length - 1) {
          bucket_id = this.buckets[i].id;
          break;
        } else {
          continue;
        }
      } else {
        bucket_id = this.buckets[i].id;
        break;
      }
    }
    return bucket_id;
  }

  private get_bucket_id(data_point: TrainingDataPoint): number {
    var bucket_id: number = 0;
    if (data_point.coordinates[0] > this.buckets[this.buckets.length - 1].id) {
      bucket_id = this.buckets[this.buckets.length - 1].id;
    } else if (data_point.coordinates[0] < this.buckets[0].id) {
      bucket_id = this.buckets[0].id;
    } else {
      for (var i = 0; i < this.buckets.length; i++) {
        if (this.buckets[i].id + this.bucket_size < data_point.coordinates[0]) {
          continue;
        } else {
          bucket_id = this.buckets[i].id;
          break;
        }
      }
    }
    return bucket_id;
  }

  public insert(params: { data_point: TrainingDataPoint }): void {
    const { data_point } = params;
    const hash = this.hash_function(data_point);
    const bucket = this.buckets.find((bucket) => bucket.id === hash);
    bucket!.data.push(data_point);
    bucket!.data.sort((a, b) => a.coordinates[1] - b.coordinates[1]);
  }

  public data_points_in_range(params: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
  }): TrainingDataPoint[] {
    const { min_x, max_x, min_y, max_y } = params;
    const data_points_in_range: TrainingDataPoint[] = [];

    const min_bucket_index = this.buckets.findIndex(
      (bucket) =>
        bucket.id === this.get_bucket_id({ coordinates: [min_x, 0], label: "" })
    );

    const max_bucket_index = this.buckets.findIndex(
      (bucket) =>
        bucket.id === this.get_bucket_id({ coordinates: [max_x, 0], label: "" })
    );

    for (let i = min_bucket_index; i <= max_bucket_index; i++) {
      const bucket = this.buckets[i];
      for (let j = 0; j < bucket.data.length; j++) {
        const data_point = bucket.data[j];
        if (
          data_point.coordinates[1] >= min_y &&
          data_point.coordinates[1] <= max_y
        ) {
          data_points_in_range.push(data_point);
        } else if (data_point.coordinates[1] > max_y) {
          break;
        } else {
          continue;
        }
      }
    }

    return data_points_in_range;
  }
}

export class VRKNN {
  private hash_table: HashTable = new HashTable();
  private training_data: TrainingDataPoint[] = [];
  private k: number;
  private radius_delta: number;
  public time_taken_to_predict: number = 0;

  constructor(params: { k: number; radius_delta: number }) {
    this.k = params.k;
    this.radius_delta = params.radius_delta;
  }

  public fit(params: { training_data: TrainingDataPoint[] }): void {
    const { training_data } = params;
    this.training_data = training_data;

    if (this.training_data.length === 0)
      throw new Error("Training data must not be empty");

    if (this.k > this.training_data.length)
      throw new Error("K must be less than the number of training data");

    this.training_data.forEach((data_point) => {
      if (data_point.coordinates.length !== 2)
        throw new Error("Data points must have 2 coordinates");
    });

    this.hash_table.create_buckets({ data: this.training_data });

    this.training_data.forEach((data_point) => {
      this.hash_table.insert({ data_point });
    });
  } // TODO

  public predict(params: { testing_data: TestDatapoint[] }): string[] {
    const { testing_data } = params;

    const predictions: string[] = [];

    if (this.training_data.length === 0)
      throw new Error("Must fit the model before predicting");

    const start_time = Date.now();

    testing_data.forEach((test_datapoint) => {
      const nearest_neighbors: TrainingDataPoint[] = [];
      const distances: [string, number][] = [];
      var x_min = test_datapoint.coordinates[0];
      var x_max = test_datapoint.coordinates[0];
      var y_min = test_datapoint.coordinates[1];
      var y_max = test_datapoint.coordinates[1];

      while (nearest_neighbors.length <= this.k) {
        x_min -= this.radius_delta;
        x_max += this.radius_delta;
        y_min -= this.radius_delta;
        y_max += this.radius_delta;

        const data_points_in_range = this.hash_table.data_points_in_range({
          max_x: x_max,
          min_x: x_min,
          max_y: y_max,
          min_y: y_min,
        });
        if (data_points_in_range.length === 0) {
          continue;
        }
        data_points_in_range.forEach((data_point) => {
          if (!nearest_neighbors.includes(data_point)) {
            nearest_neighbors.push(data_point);
          }
        });
      }
      nearest_neighbors.forEach((neighbor) => {
        distances.push([
          neighbor.label,
          distance({
            point1: [neighbor.coordinates[0], neighbor.coordinates[1]],
            point2: [
              test_datapoint.coordinates[0],
              test_datapoint.coordinates[1],
            ],
          }),
        ]);
      });

      distances.sort((a, b) => a[1] - b[1]);
      const k_nearest_neighbors = distances.slice(0, this.k);
      const labels = k_nearest_neighbors.map((neighbor) => neighbor[0]);
      predictions.push(best_of_all({ labels, algorithm: "VRKNN" }));
    });

    this.time_taken_to_predict = Date.now() - start_time;
    return predictions;
  } // TODO

  public evaluate(params: {
    predictions: string[];
    testing_labels: string[];
  }): number {
    const { predictions, testing_labels } = params;

    let correct_predictions = 0;
    predictions.forEach((prediction, index) => {
      if (prediction === testing_labels[index]) correct_predictions++;
    });
    return (correct_predictions / predictions.length) * 100;
  } // TODO
}
