// Render SVG equation assets into PNG contact sheets for visual QA.
const fs = require("fs");
const path = require("path");
const sharp = require("C:/Users/Smallopen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp");

async function main() {
  const root = path.resolve(__dirname, "..");
  const output = process.argv[2];
  if (!output) throw new Error("Usage: node render_equation_contact_sheets.js OUTPUT_DIR");
  fs.mkdirSync(output, { recursive: true });

  const groups = ["vae", "variants"];
  for (const group of groups) {
    const dir = path.join(root, "assets", "equations", group);
    const files = fs.readdirSync(dir).filter((x) => x.endsWith(".svg")).sort();
    for (let start = 0; start < files.length; start += 8) {
      const batch = files.slice(start, start + 8);
      const rendered = [];
      for (const file of batch) {
        const result = await sharp(path.join(dir, file))
          .resize({ width: 1100, withoutEnlargement: true })
          .flatten({ background: "white" })
          .png()
          .toBuffer({ resolveWithObject: true });
        rendered.push({ file, ...result });
      }
      const gap = 36;
      const labelHeight = 28;
      const width = 1200;
      const height = rendered.reduce((sum, x) => sum + x.info.height + labelHeight + gap, gap);
      let y = gap;
      const composites = [];
      for (const item of rendered) {
        const label = Buffer.from(
          `<svg width="1100" height="${labelHeight}" xmlns="http://www.w3.org/2000/svg"><text x="8" y="20" font-family="Arial" font-size="18" fill="#456">${group}/${item.file}</text></svg>`
        );
        composites.push({ input: label, left: 50, top: y });
        y += labelHeight;
        composites.push({ input: item.data, left: 50, top: y });
        y += item.info.height + gap;
      }
      const page = String(Math.floor(start / 8) + 1).padStart(2, "0");
      await sharp({ create: { width, height, channels: 3, background: "white" } })
        .composite(composites)
        .png()
        .toFile(path.join(output, `${group}-${page}.png`));
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
