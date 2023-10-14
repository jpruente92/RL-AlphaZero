import dataclasses
import json
import numpy as np

@dataclasses.dataclass
class Data:
    name: str
    values: np.ndarray

# Custom encoder for NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Create an instance of the dataclass
data_instance = Data(name="Sample", values=np.array([[1, 2],[3, 4]]))

# Serialize the dataclass to a JSON string
json_string = json.dumps(dataclasses.asdict(data_instance), cls=NumpyEncoder)
print(json_string)

# Deserialize the JSON string back to a dictionary
data = json.loads(json_string)

# Convert the list back to a NumPy array
data["values"] = np.array(data["values"])

# Convert the dictionary back to a dataclass instance
data_deserialized = Data(**data)
print(data_deserialized)
