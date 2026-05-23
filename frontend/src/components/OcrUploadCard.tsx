import type { ChangeEvent, DragEvent } from "react";

import { SectionCard } from "./SectionCard";

interface OcrUploadCardProps {
  selectedFile: File | null;
  previewUrl: string | null;
  uploading: boolean;
  warnings: string[];
  error: string | null;
  onFileChange: (file: File | null) => void;
  onUpload: () => void;
}

export function OcrUploadCard({
  selectedFile,
  previewUrl,
  uploading,
  warnings,
  error,
  onFileChange,
  onUpload,
}: OcrUploadCardProps) {
  function handleFileInputChange(event: ChangeEvent<HTMLInputElement>) {
    onFileChange(event.target.files?.[0] ?? null);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    onFileChange(event.dataTransfer.files?.[0] ?? null);
  }

  function handleDragOver(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
  }

  return (
    <SectionCard
      title="Upload CBC image"
      subtitle="Extract values into the form, then review and edit before analysis."
    >
      <div className="ocr-upload">
        <label
          className="ocr-upload__dropzone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <input
            className="ocr-upload__input"
            type="file"
            accept="image/jpeg,image/png"
            onChange={handleFileInputChange}
          />
          <span className="ocr-upload__title">
            {selectedFile ? selectedFile.name : "Choose or drop a CBC image"}
          </span>
          <span className="ocr-upload__hint">JPG, JPEG, or PNG up to 5 MB</span>
        </label>

        {previewUrl ? (
          <img className="ocr-upload__preview" src={previewUrl} alt="Selected CBC upload preview" />
        ) : null}

        {error ? <div className="ocr-upload__message ocr-upload__message--error">{error}</div> : null}

        {warnings.length > 0 ? (
          <div className="ocr-upload__message">
            <strong>Review required</strong>
            <ul>
              {warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </div>
        ) : null}

        <button
          className="secondary-button"
          type="button"
          onClick={onUpload}
          disabled={!selectedFile || uploading}
        >
          {uploading ? "Extracting..." : "Extract values"}
        </button>
      </div>
    </SectionCard>
  );
}
