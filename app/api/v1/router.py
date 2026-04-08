from fastapi import APIRouter

from app.api.v1.routes.analyses import router as analyses_router
from app.api.v1.routes.diseases import router as diseases_router
from app.api.v1.routes.indicators import router as indicators_router
from app.api.v1.routes.reference_ranges import router as reference_ranges_router

router = APIRouter(prefix="/api/v1")
router.include_router(analyses_router, tags=["analyses"])
router.include_router(indicators_router, tags=["indicators"])
router.include_router(diseases_router, tags=["diseases"])
router.include_router(reference_ranges_router, tags=["reference-ranges"])

