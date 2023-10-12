from abc import ABC, abstractmethod
import os
import zipfile


class BaseStorage(ABC):
    """Base storage class with overrides to upload different training objects to a specified storage location"""

    def __init__(self, filepath: str, object_path: str) -> None:
        self.filepath = filepath
        self.object_path = object_path

    @abstractmethod
    def upload(self) -> bool:
        """Upload the file to a storage (cloud or on-premise) to self.object_path

        Returns:
         True if the upload is successful and False otherwise
        """
        pass

    @abstractmethod
    def download(self) -> bool:
        """Download file from a storage (cloud or on-premise) to self.filepath. If file exists check last modified before downloading

        Returns:
         True if the download is successful or if the file already exists without modification, False if otherwise
        """
        pass

    @abstractmethod
    def is_modified(self) -> bool:
        """Check the last modified timestamp in UTC and compare it with the local file

        Returns:
            True if the timestamps don't match otherwise False
        """
        pass
