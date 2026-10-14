import { Router, type IRouter } from "express";
import healthRouter from "./health";
import fraudRouter from "./fraud";

const router: IRouter = Router();

router.use(healthRouter);
router.use(fraudRouter);

export default router;
