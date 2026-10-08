from pathlib import Path
import json
import zipfile
import io

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import app as graph_app


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"
IMAGES_DIR = BASE_DIR / "images"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ForgeBlog AI API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class GenerateRequest(BaseModel):
    topic: str
    as_of: str
    recency_days: int = 7


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_blog_files():
    """
    Find generated markdown blogs in the project root.
    """
    files = []

    for file in BASE_DIR.glob("*.md"):
        if file.name.lower() in {
            "readme.md",
            "requirements.md",
        }:
            continue

        files.append(file)

    files.sort(
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    return files


def safe_filename(filename: str) -> Path:
    """
    Prevent path traversal and make sure the requested
    file exists inside the project directory.
    """
    filename = Path(filename).name
    path = BASE_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Blog file not found",
        )

    return path


# ============================================================
# API: LIST BLOGS
# ============================================================

@app.get("/api/blogs")
def list_blogs():
    blogs = []

    for file in get_blog_files():
        blogs.append(
            {
                "filename": file.name,
                "title": file.stem,
                "modified": file.stat().st_mtime,
            }
        )

    return {
        "blogs": blogs
    }


# ============================================================
# API: READ BLOG
# ============================================================

@app.get("/api/blogs/{filename}")
def read_blog(filename: str):

    file_path = safe_filename(filename)

    if file_path.suffix.lower() != ".md":
        raise HTTPException(
            status_code=400,
            detail="Only markdown files are allowed",
        )

    try:
        content = file_path.read_text(
            encoding="utf-8"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    return {
        "filename": file_path.name,
        "title": file_path.stem,
        "markdown": content,
    }


# ============================================================
# API: GENERATE BLOG
# ============================================================

@app.post("/api/generate")
def generate_blog(request: GenerateRequest):

    topic = request.topic.strip()

    if not topic:
        raise HTTPException(
            status_code=400,
            detail="Topic is required",
        )

    inputs = {
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": request.as_of,
        "recency_days": request.recency_days,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    def event_stream():

        try:

            yield (
                "event: started\n"
                f"data: {json.dumps({'topic': topic})}\n\n"
            )

            final_state = {}

            for update in graph_app.stream(
                inputs,
                stream_mode="updates",
            ):

                # ------------------------------------------------
                # Collect updates
                # ------------------------------------------------

                if isinstance(update, dict):

                    for node_name, node_state in update.items():

                        if isinstance(node_state, dict):
                            final_state.update(node_state)

                        progress_data = {
                            "node": node_name,
                            "state": node_state,
                        }

                        yield (
                            "event: progress\n"
                            f"data: {json.dumps(progress_data, default=str)}\n\n"
                        )

            # ----------------------------------------------------
            # Determine generated markdown
            # ----------------------------------------------------

            markdown = final_state.get("final")

            if not markdown:
                markdown = final_state.get("merged_md")

            # ----------------------------------------------------
            # Try to find generated markdown file
            # ----------------------------------------------------

            blog_files = get_blog_files()

            filename = None

            if blog_files:
                filename = blog_files[0].name

            # ----------------------------------------------------
            # Plan
            # ----------------------------------------------------

            plan = final_state.get("plan")

            if hasattr(plan, "model_dump"):
                plan = plan.model_dump()

            # ----------------------------------------------------
            # Evidence
            # ----------------------------------------------------

            evidence = final_state.get(
                "evidence",
                [],
            )

            if hasattr(evidence, "model_dump"):
                evidence = evidence.model_dump()

            elif isinstance(evidence, list):

                converted_evidence = []

                for item in evidence:

                    if hasattr(item, "model_dump"):
                        converted_evidence.append(
                            item.model_dump()
                        )
                    else:
                        converted_evidence.append(item)

                evidence = converted_evidence

            # ----------------------------------------------------
            # Images
            # ----------------------------------------------------

            image_specs = final_state.get(
                "image_specs",
                [],
            )

            converted_images = []

            if isinstance(image_specs, list):

                for item in image_specs:

                    if hasattr(item, "model_dump"):
                        converted_images.append(
                            item.model_dump()
                        )
                    else:
                        converted_images.append(item)

            image_specs = converted_images

            # ----------------------------------------------------
            # Complete response
            # ----------------------------------------------------

            result = {
                "filename": filename,
                "title": (
                    Path(filename).stem
                    if filename
                    else topic
                ),
                "markdown": markdown or "",
                "plan": plan,
                "evidence": evidence,
                "images": image_specs,
            }

            yield (
                "event: complete\n"
                f"data: {json.dumps(result, default=str)}\n\n"
            )

        except Exception as e:

            error_data = {
                "error": str(e)
            }

            yield (
                "event: error\n"
                f"data: {json.dumps(error_data)}\n\n"
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# API: SERVE IMAGE
# ============================================================

@app.get("/api/images/{filename}")
def get_image(filename: str):

    filename = Path(filename).name

    image_path = IMAGES_DIR / filename

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Image not found",
        )

    return FileResponse(image_path)


# ============================================================
# API: DOWNLOAD MARKDOWN
# ============================================================

@app.get("/api/download/markdown/{filename}")
def download_markdown(filename: str):

    file_path = safe_filename(filename)

    if file_path.suffix.lower() != ".md":
        raise HTTPException(
            status_code=400,
            detail="Only markdown files are allowed",
        )

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="text/markdown",
    )


# ============================================================
# API: DOWNLOAD BUNDLE
# ============================================================

@app.get("/api/download/bundle/{filename}")
def download_bundle(filename: str):

    markdown_path = safe_filename(filename)

    if markdown_path.suffix.lower() != ".md":
        raise HTTPException(
            status_code=400,
            detail="Only markdown files are allowed",
        )

    memory_file = io.BytesIO()

    with zipfile.ZipFile(
        memory_file,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zip_file:

        # Add markdown
        zip_file.write(
            markdown_path,
            arcname=markdown_path.name,
        )

        # Add images if they exist
        if IMAGES_DIR.exists():

            for image in IMAGES_DIR.iterdir():

                if image.is_file():

                    zip_file.write(
                        image,
                        arcname=f"images/{image.name}",
                    )

    memory_file.seek(0)

    return StreamingResponse(
        memory_file,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{markdown_path.stem}_bundle.zip"'
            )
        },
    )


# ============================================================
# API: DOWNLOAD ALL IMAGES
# ============================================================

@app.get("/api/download/images/{filename}")
def download_images(filename: str):

    # Verify blog exists
    safe_filename(filename)

    memory_file = io.BytesIO()

    with zipfile.ZipFile(
        memory_file,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zip_file:

        if IMAGES_DIR.exists():

            for image in IMAGES_DIR.iterdir():

                if image.is_file():

                    zip_file.write(
                        image,
                        arcname=image.name,
                    )

    memory_file.seek(0)

    return StreamingResponse(
        memory_file,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{Path(filename).stem}_images.zip"'
            )
        },
    )


# ============================================================
# FRONTEND
# IMPORTANT: THIS MUST BE LAST
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True,
    ),
    name="frontend",
)