import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():

    return


@app.cell
def _():
    import os

    import polars as pl

    return os, pl


@app.cell
def _(os):
    from enum import StrEnum

    breast_level_annotations_file, findings_annotations_file, metadata_file = sorted(
        filter(lambda x: x.endswith(".csv"), os.listdir("data/"))
    )[:3]

    class AnnotationFiles(StrEnum):
        BreastLevelAnnotations = os.path.join("data/", breast_level_annotations_file)
        FindingsAnnotations = os.path.join("data/", findings_annotations_file)
        Metadata = os.path.join(
            "data/",
            metadata_file,
        )

    return (AnnotationFiles,)


@app.cell
def _(AnnotationFiles, pl):
    df_bla = pl.read_csv(AnnotationFiles.BreastLevelAnnotations)
    df_fa = pl.read_csv(AnnotationFiles.FindingsAnnotations)
    df_metadata = pl.read_csv(
        AnnotationFiles.Metadata,
        schema_overrides={
            "Window Center": pl.String,
            "Window Width": pl.String,
        },
    )
    return


if __name__ == "__main__":
    app.run()
