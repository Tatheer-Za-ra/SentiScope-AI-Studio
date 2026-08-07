import codecs
import csv
import io
from typing import BinaryIO, Dict, Generator, List, Optional, Tuple, Union
import pandas as pd


class CSVParsingError(Exception):
    """Custom exception raised when structural or parsing validation fails."""

    def __init__(self, message: str, line_number: Optional[int] = None, row_index: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.line_number = line_number
        self.row_index = row_index


class RobustCSVParser:
    """Production-grade CSV parser handling Level 1 Structural & Parsing edge cases:
    - 0-Byte Empty Files
    - Streaming & Large File Handling
    - Byte-Order Mark (BOM) Stripping (UTF-8, UTF-16)
    - Encoding Mismatch Detection (UTF-8, Windows-1252, ISO-8859-1)
    - Mixed Line Endings (\r\n, \n, \r)
    - Delimiter Auto-Detection (,, ;, \t, |)
    - Quoted Fields containing Delimiters
    - Dangling / Unescaped Quotes Handling
    - Multiline Cells Support
    - Column Count Mismatches (Ragged Rows)
    - Trailing Commas & Phantom Blank Lines Cleanup
    """

    SUPPORTED_DELIMITERS = [",", ";", "\t", "|"]
    MAX_PREVIEW_BYTES = 64 * 1024  # 64 KB for header and delimiter sniffing

    def __init__(
        self,
        file_stream: BinaryIO,
        max_file_size_bytes: int = 50 * 1024 * 1024,  # 50MB ceiling
        max_row_length_bytes: int = 10 * 1024 * 1024,  # 10MB per row
        expected_headers: Optional[List[str]] = None,
        strict_column_count: bool = False,
    ):
        self.file_stream = file_stream
        self.max_file_size_bytes = max_file_size_bytes
        self.max_row_length_bytes = max_row_length_bytes
        self.expected_headers = expected_headers
        self.strict_column_count = strict_column_count

        self.detected_encoding: str = "utf-8"
        self.detected_delimiter: str = ","
        self.headers: List[str] = []
        self.total_file_bytes: int = 0
        self.skipped_blank_rows: int = 0

    def parse_stream(self) -> Generator[Dict[str, str], None, None]:
        """Streams valid CSV rows as dictionaries mapping header -> cell value."""
        self.file_stream.seek(0, io.SEEK_END)
        self.total_file_bytes = self.file_stream.tell()
        self.file_stream.seek(0)

        if self.total_file_bytes == 0:
            raise CSVParsingError("The uploaded CSV file is empty (0 bytes). Please upload a file with data.")

        if self.total_file_bytes > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            actual_mb = self.total_file_bytes / (1024 * 1024)
            raise CSVParsingError(f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.2f} MB.")

        # Detect BOM & Encodings
        preview_bytes = self.file_stream.read(min(self.MAX_PREVIEW_BYTES, self.total_file_bytes))
        self.file_stream.seek(0)

        self.detected_encoding = self._detect_encoding(preview_bytes)

        text_stream = io.TextIOWrapper(
            self.file_stream,
            encoding=self.detected_encoding,
            errors="replace",
            newline=None,
        )

        preview_text = self._read_preview_text(preview_bytes, self.detected_encoding)
        self.detected_delimiter = self._detect_delimiter(preview_text)

        # Set higher field size limit safely
        try:
            csv.field_size_limit(self.max_row_length_bytes)
        except Exception:
            pass

        reader = csv.reader(
            text_stream,
            delimiter=self.detected_delimiter,
            quotechar='"',
            skipinitialspace=True,
            strict=False,
        )

        try:
            raw_headers = next(reader)
        except StopIteration:
            raise CSVParsingError("CSV file has no header row.")
        except csv.Error as e:
            raise CSVParsingError(f"Failed to parse CSV header row: {str(e)}")

        self.headers = self._clean_headers(raw_headers)

        if not self.headers or all(h == "" for h in self.headers):
            raise CSVParsingError("CSV file header row contains no valid column names.")

        expected_count = len(self.headers)

        if self.expected_headers:
            missing = [h for h in self.expected_headers if h not in self.headers]
            if missing:
                raise CSVParsingError(f"Missing required column headers: {', '.join(missing)}")

        row_index = 0
        for raw_row in reader:
            row_index += 1

            if not raw_row or (len(raw_row) == 1 and str(raw_row[0]).strip() == ""):
                self.skipped_blank_rows += 1
                continue

            # Handle ragged / mismatched column count
            if len(raw_row) != expected_count:
                if len(raw_row) < expected_count:
                    if not self.strict_column_count:
                        raw_row.extend([""] * (expected_count - len(raw_row)))
                    else:
                        raise CSVParsingError(
                            f"Row {row_index} has {len(raw_row)} columns, expected {expected_count}.",
                            row_index=row_index,
                        )
                elif len(raw_row) > expected_count:
                    extra_fields = raw_row[expected_count:]
                    if all(str(f).strip() == "" for f in extra_fields):
                        raw_row = raw_row[:expected_count]
                    else:
                        if not self.strict_column_count:
                            raw_row = raw_row[:expected_count]
                        else:
                            raise CSVParsingError(
                                f"Row {row_index} has {len(raw_row)} columns, exceeding header count of {expected_count}.",
                                row_index=row_index,
                            )

            cleaned_row = {
                header: str(raw_row[i]).strip() for i, header in enumerate(self.headers)
            }

            yield cleaned_row

    def _detect_encoding(self, preview_bytes: bytes) -> str:
        if preview_bytes.startswith(codecs.BOM_UTF8):
            return "utf-8-sig"
        elif preview_bytes.startswith(codecs.BOM_UTF16_LE):
            return "utf-16-le"
        elif preview_bytes.startswith(codecs.BOM_UTF16_BE):
            return "utf-16-be"

        try:
            preview_bytes.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            pass

        try:
            import charset_normalizer
            result = charset_normalizer.from_bytes(preview_bytes).best()
            if result and result.encoding:
                return result.encoding
        except ImportError:
            pass

        return "windows-1252"

    def _read_preview_text(self, preview_bytes: bytes, encoding: str) -> str:
        try:
            return preview_bytes.decode(encoding, errors="ignore")
        except Exception:
            return preview_bytes.decode("latin-1", errors="ignore")

    def _detect_delimiter(self, sample_text: str) -> str:
        first_line = sample_text.splitlines()[0] if sample_text.splitlines() else ""
        if not first_line:
            return ","

        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text, delimiters=",".join(self.SUPPORTED_DELIMITERS))
            return dialect.delimiter
        except csv.Error:
            counts = {d: first_line.count(d) for d in self.SUPPORTED_DELIMITERS}
            best_delimiter = max(counts, key=counts.get)
            return best_delimiter if counts[best_delimiter] > 0 else ","

    def _clean_headers(self, raw_headers: List[str]) -> List[str]:
        cleaned = []
        for idx, header in enumerate(raw_headers):
            h = str(header).lstrip("\ufeff").strip().strip('"').strip("'")
            if not h:
                h = f"unnamed_column_{idx + 1}"
            cleaned.append(h)
        return cleaned


def parse_csv_file(uploaded_file, strict_column_count: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Helper to parse uploaded Streamlit file into DataFrame with parsing metadata."""
    parser = RobustCSVParser(uploaded_file, strict_column_count=strict_column_count)
    rows = list(parser.parse_stream())
    if not rows:
        return None, {
            "encoding": parser.detected_encoding,
            "delimiter": parser.detected_delimiter,
            "headers": parser.headers,
            "total_bytes": parser.total_file_bytes,
            "skipped_blank_rows": parser.skipped_blank_rows,
        }
    
    df = pd.DataFrame(rows)
    meta = {
        "encoding": parser.detected_encoding,
        "delimiter": parser.detected_delimiter,
        "headers": parser.headers,
        "total_bytes": parser.total_file_bytes,
        "skipped_blank_rows": parser.skipped_blank_rows,
    }
    return df, meta
