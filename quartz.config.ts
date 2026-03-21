import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

const config: QuartzConfig = {
  configuration: {
    pageTitle: "notes.quant",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "en-US",
    baseUrl: "geilerloui.github.io/notes-quant",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Lora",
        body: "Lora",
        code: "JetBrains Mono",
      },
      colors: {
        lightMode: {
          light: "#f7f5f0",
          lightgray: "#eeecea",
          gray: "#8a8784",
          darkgray: "#4a4845",
          dark: "#1a1917",
          secondary: "#c0392b",
          tertiary: "#e8b4b0",
          highlight: "rgba(192, 57, 43, 0.08)",
          textHighlight: "#e8b4b088",
        },
        darkMode: {
          light: "#1a1917",
          lightgray: "#2c2c2a",
          gray: "#5f5e5a",
          darkgray: "#d3d1c7",
          dark: "#f1efe8",
          secondary: "#e8b4b0",
          tertiary: "#c0392b",
          highlight: "rgba(232, 180, 176, 0.08)",
          textHighlight: "#c0392b88",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      Plugin.CustomOgImages(),
    ],
  },
}
export default config