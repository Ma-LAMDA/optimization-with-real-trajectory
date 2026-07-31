#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const EXPERIMENT_ROOT = path.resolve(SCRIPT_DIR, "..");
const REPOSITORY_ROOT = path.resolve(EXPERIMENT_ROOT, "..", "..");
const REPORTS_DIR = path.join(EXPERIMENT_ROOT, "results", "reports");
const RUN_DIR = path.join(EXPERIMENT_ROOT, "results", "runs", "fullaccess");
const DATASET_PATH = path.join(
  REPOSITORY_ROOT,
  "data",
  "simulation",
  "train_0629.jsonl",
);
const PREVIEW_DIR = path.join(
  EXPERIMENT_ROOT,
  "runtime",
  "spreadsheet_artifact",
);
const CASE_IDS = [13, 14, 17, 18, 87, 88, 91, 92, 93, 94];
const REPEATS = 10;

function sha256Text(text) {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

function canonicalFaultSet(value) {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new TypeError("fault result must be a JSON array of strings");
  }
  return [...new Set(value)].sort((left, right) =>
    left.localeCompare(right, "zh-CN"),
  );
}

function parseFinalAnswer(text) {
  const matches = [
    ...text.matchAll(/<result>\s*([\s\S]*?)\s*<\/result>/g),
  ];
  if (matches.length !== 1) {
    return {
      ok: false,
      status: "wrapper_count_" + matches.length,
      faults: [],
    };
  }
  try {
    const parsed = JSON.parse(matches[0][1]);
    return {
      ok: true,
      status: "parsed",
      faults: canonicalFaultSet(parsed),
    };
  } catch (error) {
    return {
      ok: false,
      status: "json_error: " + String(error.message || error),
      faults: [],
    };
  }
}

function equalArrays(left, right) {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function csvCell(value) {
  if (value === null || value === undefined) {
    return "";
  }
  const text = String(value);
  return /[",\r\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
}

function csvText(matrix) {
  return (
    "\uFEFF" +
    matrix.map((row) => row.map(csvCell).join(",")).join("\r\n") +
    "\r\n"
  );
}

function joinFaults(faults) {
  return faults.join(" | ");
}

await fs.mkdir(REPORTS_DIR, { recursive: true });
await fs.mkdir(PREVIEW_DIR, { recursive: true });

const datasetText = await fs.readFile(DATASET_PATH, "utf8");
const datasetRows = datasetText
  .split(/\r?\n/)
  .filter((line) => line.trim())
  .map((line) => JSON.parse(line));
const rowById = new Map(datasetRows.map((row) => [row.id, row]));
const manifestText = await fs.readFile(
  path.join(RUN_DIR, "manifest.json"),
  "utf8",
);
const manifest = JSON.parse(manifestText);
const runBySlot = new Map(
  manifest.runs.map((run) => [
    String(run.case_id) + ":" + String(run.repeat_index),
    run,
  ]),
);

const detailRecords = [];
for (const caseId of CASE_IDS) {
  const source = rowById.get(caseId);
  if (!source) {
    throw new Error("missing source row for case " + caseId);
  }
  const gold = canonicalFaultSet(JSON.parse(source.answer));
  for (let repeat = 1; repeat <= REPEATS; repeat += 1) {
    const run = runBySlot.get(String(caseId) + ":" + String(repeat));
    if (!run || run.status !== "succeeded") {
      throw new Error(
        "slot is not succeeded: q" +
          String(caseId).padStart(4, "0") +
          "_r" +
          String(repeat).padStart(2, "0"),
      );
    }
    const slotName =
      "q" +
      String(caseId).padStart(4, "0") +
      "_r" +
      String(repeat).padStart(2, "0");
    const attemptName =
      "attempt_" + String(run.successful_attempt).padStart(3, "0");
    const attemptDir = path.join(RUN_DIR, slotName, attemptName);
    const finalPath = path.join(attemptDir, "final_answer.txt");
    const finalText = await fs.readFile(finalPath, "utf8");
    const metadata = JSON.parse(
      await fs.readFile(path.join(attemptDir, "metadata.json"), "utf8"),
    );
    const parsed = parseFinalAnswer(finalText);
    const correct = parsed.ok && equalArrays(parsed.faults, gold);
    detailRecords.push({
      caseId,
      repeat,
      slotName,
      attemptIndex: run.successful_attempt,
      threadId: metadata.thread_id || "",
      verdict: parsed.ok ? (correct ? "正确" : "错误") : "解析失败",
      parseStatus: parsed.status,
      predicted: joinFaults(parsed.faults),
      expected: joinFaults(gold),
      finalAnswerPath: path.relative(EXPERIMENT_ROOT, finalPath).replaceAll("\\", "/"),
      finalAnswerSha256: sha256Text(finalText),
      correct,
      parseFailed: !parsed.ok,
    });
  }
}

const summaryHeader = [
  "题号",
  "总轨迹数",
  "正确数",
  "错误数",
  "准确率(%)",
  "正确轮次",
  "错误轮次",
  "解析失败数",
  "标准答案",
];
const summaryRows = [];
for (const caseId of CASE_IDS) {
  const records = detailRecords.filter((record) => record.caseId === caseId);
  const correct = records.filter((record) => record.correct);
  const incorrect = records.filter((record) => !record.correct);
  const parseFailures = records.filter((record) => record.parseFailed);
  const expected = canonicalFaultSet(JSON.parse(rowById.get(caseId).answer));
  summaryRows.push([
    caseId,
    records.length,
    correct.length,
    incorrect.length,
    Number(((correct.length / records.length) * 100).toFixed(2)),
    correct.map((record) => record.repeat).join(" "),
    incorrect.map((record) => record.repeat).join(" "),
    parseFailures.length,
    joinFaults(expected),
  ]);
}
const totalRuns = detailRecords.length;
const totalCorrect = detailRecords.filter((record) => record.correct).length;
const totalParseFailures = detailRecords.filter(
  (record) => record.parseFailed,
).length;
const totalRow = [
  "总计",
  totalRuns,
  totalCorrect,
  totalRuns - totalCorrect,
  Number(((totalCorrect / totalRuns) * 100).toFixed(2)),
  "",
  "",
  totalParseFailures,
  "10题；故障集合精确匹配，忽略列表顺序",
];
const summaryMatrix = [summaryHeader, ...summaryRows, totalRow];

const detailHeader = [
  "题号",
  "轮次",
  "槽位",
  "成功attempt",
  "thread_id",
  "判定",
  "解析状态",
  "模型答案",
  "标准答案",
  "final_answer相对路径",
  "final_answer_sha256",
];
const detailMatrix = [
  detailHeader,
  ...detailRecords.map((record) => [
    record.caseId,
    record.repeat,
    record.slotName,
    record.attemptIndex,
    record.threadId,
    record.verdict,
    record.parseStatus,
    record.predicted,
    record.expected,
    record.finalAnswerPath,
    record.finalAnswerSha256,
  ]),
];

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("各题准确率");
const detailSheet = workbook.worksheets.add("逐轨迹明细");
summarySheet.showGridLines = false;
detailSheet.showGridLines = false;
summarySheet.freezePanes.freezeRows(1);
detailSheet.freezePanes.freezeRows(1);

const summaryValues = summaryMatrix.map((row) => [...row]);
for (let index = 1; index <= CASE_IDS.length; index += 1) {
  summaryValues[index][3] = null;
  summaryValues[index][4] = null;
}
summaryValues[summaryValues.length - 1][1] = null;
summaryValues[summaryValues.length - 1][2] = null;
summaryValues[summaryValues.length - 1][3] = null;
summaryValues[summaryValues.length - 1][4] = null;
summaryValues[summaryValues.length - 1][7] = null;
summarySheet
  .getRangeByIndexes(0, 0, summaryValues.length, summaryHeader.length)
  .values = summaryValues;

summarySheet.getRange("D2").formulas = [["=B2-C2"]];
summarySheet.getRange("D2:D11").fillDown();
summarySheet.getRange("E2").formulas = [["=IF(B2=0,0,C2/B2*100)"]];
summarySheet.getRange("E2:E11").fillDown();
summarySheet.getRange("B12").formulas = [["=SUM(B2:B11)"]];
summarySheet.getRange("C12").formulas = [["=SUM(C2:C11)"]];
summarySheet.getRange("D12").formulas = [["=SUM(D2:D11)"]];
summarySheet.getRange("E12").formulas = [["=IF(B12=0,0,C12/B12*100)"]];
summarySheet.getRange("H12").formulas = [["=SUM(H2:H11)"]];

summarySheet.getRange("A1:I1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
summarySheet.getRange("A12:I12").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  borders: { preset: "doubleBottom", style: "medium", color: "#5B9BD5" },
};
summarySheet.getRange("A2:H12").format.horizontalAlignment = "center";
summarySheet.getRange("I2:I12").format.wrapText = true;
summarySheet.getRange("E2:E12").format.numberFormat = "0.00";
summarySheet.getRange("A1:I12").format.borders = {
  preset: "inside",
  style: "thin",
  color: "#D9E2F3",
};
summarySheet.getRange("A1:A12").format.columnWidth = 10;
summarySheet.getRange("B1:E12").format.columnWidth = 12;
summarySheet.getRange("F1:H12").format.columnWidth = 20;
summarySheet.getRange("I1:I12").format.columnWidth = 42;
summarySheet.getRange("A1:I12").format.autofitRows();

detailSheet
  .getRangeByIndexes(0, 0, detailMatrix.length, detailHeader.length)
  .values = detailMatrix;
detailSheet.getRange("A1:K1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
detailSheet.getRange("A2:G101").format.horizontalAlignment = "center";
detailSheet.getRange("H2:J101").format.wrapText = true;
detailSheet.getRange("A1:K101").format.borders = {
  preset: "inside",
  style: "thin",
  color: "#E7E6E6",
};
detailSheet.getRange("A1:B101").format.columnWidth = 9;
detailSheet.getRange("C1:D101").format.columnWidth = 16;
detailSheet.getRange("E1:E101").format.columnWidth = 38;
detailSheet.getRange("F1:G101").format.columnWidth = 14;
detailSheet.getRange("H1:I101").format.columnWidth = 38;
detailSheet.getRange("J1:J101").format.columnWidth = 48;
detailSheet.getRange("K1:K101").format.columnWidth = 68;
detailSheet.getRange("A1:K101").format.autofitRows();
detailSheet.getRange("F2:F101").conditionalFormats.add("containsText", {
  text: "正确",
  format: { fill: "#E2F0D9", font: { color: "#006100" } },
});
detailSheet.getRange("F2:F101").conditionalFormats.add("containsText", {
  text: "错误",
  format: { fill: "#FCE4D6", font: { color: "#9C0006" } },
});
detailSheet.getRange("F2:F101").conditionalFormats.add("containsText", {
  text: "解析失败",
  format: { fill: "#FFF2CC", font: { color: "#9C6500" } },
});

await workbook.recalculate();
const summaryInspection = await workbook.inspect({
  kind: "table",
  sheetId: "各题准确率",
  range: "A1:I12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 9,
  maxChars: 12000,
});
const detailInspection = await workbook.inspect({
  kind: "table",
  sheetId: "逐轨迹明细",
  range: "A1:K12",
  include: "values",
  tableMaxRows: 12,
  tableMaxCols: 11,
  maxChars: 12000,
});
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});

const summaryPreview = await workbook.render({
  sheetName: "各题准确率",
  range: "A1:I12",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  path.join(PREVIEW_DIR, "accuracy_summary.png"),
  new Uint8Array(await summaryPreview.arrayBuffer()),
);
const detailPreview = await workbook.render({
  sheetName: "逐轨迹明细",
  range: "A1:K18",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(PREVIEW_DIR, "accuracy_detail.png"),
  new Uint8Array(await detailPreview.arrayBuffer()),
);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(REPORTS_DIR, "准确率统计审计.xlsx"));
await fs.writeFile(
  path.join(REPORTS_DIR, "各题准确率统计.csv"),
  csvText(summaryMatrix),
  "utf8",
);
await fs.writeFile(
  path.join(REPORTS_DIR, "逐轨迹判分明细.csv"),
  csvText(detailMatrix),
  "utf8",
);

const evaluation = {
  schema_version: "codex-ip-trajectory-accuracy.v1",
  evaluated_at: new Date().toISOString(),
  scoring_rule: {
    source: "data/simulation/train_0629.jsonl answer field",
    parser: "exactly one <result> wrapper containing a JSON array of strings",
    comparison: "exact fault-set match; list order ignored",
    duplicates: "collapsed as set members",
    missing_extra_or_wrong_fault: "incorrect",
    parse_failure: "incorrect and counted separately",
  },
  source_hashes: {
    dataset_sha256: sha256Text(datasetText),
    manifest_sha256: sha256Text(manifestText),
  },
  case_ids: CASE_IDS,
  repeats_per_case: REPEATS,
  total_runs: totalRuns,
  correct_runs: totalCorrect,
  incorrect_runs: totalRuns - totalCorrect,
  parse_failures: totalParseFailures,
  accuracy_percent: Number(((totalCorrect / totalRuns) * 100).toFixed(2)),
  summary: summaryRows.map((row) => ({
    case_id: row[0],
    total: row[1],
    correct: row[2],
    incorrect: row[3],
    accuracy_percent: row[4],
    correct_repeats: row[5],
    incorrect_repeats: row[6],
    parse_failures: row[7],
    expected: row[8],
  })),
  files: {
    summary_csv: "results/reports/各题准确率统计.csv",
    detail_csv: "results/reports/逐轨迹判分明细.csv",
    audit_xlsx: "results/reports/准确率统计审计.xlsx",
  },
};
await fs.writeFile(
  path.join(REPORTS_DIR, "accuracy_evaluation.json"),
  JSON.stringify(evaluation, null, 2) + "\n",
  "utf8",
);

console.log(JSON.stringify(evaluation, null, 2));
console.log("SUMMARY_INSPECTION");
console.log(summaryInspection.ndjson);
console.log("DETAIL_INSPECTION");
console.log(detailInspection.ndjson);
console.log("FORMULA_ERRORS");
console.log(formulaErrors.ndjson);
