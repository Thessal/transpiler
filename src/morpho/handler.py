from typing import Dict, List, Union
import re
from morpho.util import load_text


class BaseHandler:
    def __init__(self):
        self.required_fields = {"data_type", "last_validated"}
        self.optional_fields = {"serialized", "embedding", "additional_info"}
        self.constraints: Dict[str, Union[List[str], re.Pattern]] = {
            "last_validated": re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"),
            # 0.0 to 1.0
            "confidence_level": re.compile(r"^(0(\.\d+)?|1(\.0+)?)$"),
        }

    def validate(self, metadata: Dict):
        for field in self.required_fields:
            if not (field in metadata):
                raise ValueError(f"Missing field {field} in (unknown json) for {metadata['path']}")
        for field, value in metadata.items():
            if not (
                (field in self.required_fields) 
                or (field in self.optional_fields)
                ):
                raise ValueError(f"Unknown field {field} in (unknown json) for in {metadata['path']}")
            if field in self.constraints:
                target = self.constraints[field]
                if isinstance(target, list):
                    if not (value in target):
                        raise ValueError(f"Unknown option {field}={value} in (unknown json) for {metadata['path']}")
                else:
                    if not bool(target.match(value)):
                        raise ValueError(f"Unknown value {field}={value} in (unknown json) for {metadata['path']}")

    def serialize(self, metadata: Dict) -> Dict:
        self.validate(metadata)
        return metadata


class DocumentHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.required_fields |= {"path", "data_source", "environment", "agent"}
        self.optional_fields |= {"keywords",
                                 "related_path", "confidence_level", }
        self.constraints |= {
            "data_type": ["document"],
            "path": re.compile(r"^([\w\-\.\/\\\ ]+)\.(txt|TXT)$"),
            "data_source": ["web-conversation", "web-article", "book", "academic-article", "other"],
            "environment": ["research", "paper-trading", "live"],
            # e.g. human#e4bfd7
            "agent": re.compile(r"^(human|machine)#?([a-fA-F0-9]{6})$"),
            # comma separated tags
            "keywords": re.compile(r"^([\w-]+)(,\s*[\w-]+)*$"),
            "related_path": re.compile(r"^([\w\-\.\/\\\ ]+)\.(json)$"),
        }

    def serialize(self, metadata: Dict):
        text = load_text(metadata["path"])
        text = text.replace("\n", "  ")
        metadata["serialized"] = text
        return super().serialize(metadata)


class CodeHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.required_fields |= {"path", "environment", "agent"}
        self.optional_fields |= {"keywords",
                                 "related_path", "confidence_level", }
        self.constraints |= {
            "data_type": ["code"],
            "path": re.compile(r"^([\w\-\.\/\\\ ]+)\.(py|mq4|mq5|bf)$"),
            "environment": ["research", "paper-trading", "live"],
            # e.g. human#e4bfd7
            "agent": re.compile(r"^(human|machine)#?([a-fA-F0-9]{6})$"),
            # comma separated tags
            "keywords": re.compile(r"^([\w-]+)(,\s*[\w-]+)*$"),
            "related_path": re.compile(r"^([\w\-\.\/\\\ ]+)\.(json)$"),
        }

    def serialize(self, metadata):
        text = load_text(metadata["path"])
        text = "\n".join(text.split())
        metadata["serialized"] = text
        return super().serialize(metadata)


class SummaryHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.required_fields |= {"path", "model", "prompt_path"}
        self.optional_fields |= {"original_path"}
        self.constraints |= {
            "data_type": ["summary"],
            "path": re.compile(r"^([\w\-\.\/\\\ ]+)\.(txt)$"),
            "prompt_path": re.compile(r"^([\w\-\.\/\\\ ]+)\.(txt)$"),
            "original_path": re.compile(r"^([\w\-\.\/\\\ ]+)\.(json)$"),
        }

    def serialize(self, metadata: Dict):
        metadata["serialized"] = load_text(metadata["path"])
        return super().serialize(metadata)

# class YoutubeHandler(BaseHandler):
#    # summary

# class ImageHandler(BaseHandler)
#    # text description, base64 etc


handlers = {
    "document": DocumentHandler(),
    "code": CodeHandler(),
    "summary": SummaryHandler(),
}
