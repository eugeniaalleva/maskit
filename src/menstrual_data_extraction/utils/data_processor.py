from typing import *
import pandas as pd

class MenstrualDataProcessor:
    def __init__(self, labels: Optional[Sequence[Any]] = None):
        if labels is not None:
            self.labels = labels

    @property
    def labels(self) -> List[Any]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return self._labels

    @labels.setter
    def labels(self, labels: Sequence[Any]):
        if labels is not None:
            self._labels = labels
            self._label_mapping = {k: i for (i, k) in enumerate(labels)}

    @property
    def label_mapping(self) -> Dict[Any, int]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return self._label_mapping

    @label_mapping.setter
    def label_mapping(self, label_mapping: Mapping[Any, int]):
        self._labels = [item[0] for item in sorted(label_mapping.items(), key=lambda item: item[1])]
        self._label_mapping = label_mapping

    @property
    def id2label(self) -> Dict[int, Any]:
        if not hasattr(self, "_labels"):
            raise ValueError("DataProcessor doesn't set labels or label_mapping yet")
        return {i: k for (i, k) in enumerate(self._labels)}
    
    def get_label_id(self, label: Any) -> int:
        """get label id of the corresponding label

        Args:
            label: label in dataset

        Returns:
            int: the index of label
        """
        return self.label_mapping[label] if label is not None else None

    def get_labels(self) -> List[Any]:
        """get labels of the dataset

        Returns:
            List[Any]: labels of the dataset
        """
        return self.labels

    def get_multimask_examples(self, data_dir, tasks, note_text_columnt = "NOTE_TEXT"):
        df_menstruates = pd.read_csv(data_dir)
        texts = []
        labels = {}

        for _, row in df_menstruates.iterrows():
            note_text =  row[note_text_columnt]
            
            for task in tasks.keys():
                task_config = tasks[task]
                self.labels = task_config.labels
                label = row[task_config.characteristic]

                if task not in labels:
                    labels[task] = [self.get_label_id(label)]
                else:
                    labels[task].append(self.get_label_id(label))
            
            texts.append(note_text)

        return texts, labels