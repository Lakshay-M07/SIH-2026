import express from "express";

const app = express();
app.use(express.json());

app.get("/", (_req, res) => {
  res.json({ service: "SIH26162 backend", status: "initialized" });
});

app.listen(process.env.PORT || 5000, () => {
  console.log(`Backend running on port ${process.env.PORT || 5000}`);
});
