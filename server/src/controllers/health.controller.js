export const getHealth = (_req, res) => {
  res.json({
    status: "ok",
    service: "SIH26162 backend",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    integrations: {
      firebase: Boolean(process.env.FIREBASE_PROJECT_ID),
      mlService: "ready-for-handover",
    },
  });
};
