import json
import time

import pandas as pd
import requests


def scrape_huxiu_success():
    # 接口地址
    url = "https://api-data-mini.huxiu.com/hxgpt/agent/ai-product-daily/v3/detail-list"

    # 请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Origin": "https://xiuxiu.huxiu.com",
        "Referer": "https://xiuxiu.huxiu.com/",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    # 参数
    payload = {
        "date": "2026-01-26",
        "platform": "www",
        "page_num": "1",
        "page_size": "50",
    }

    print(f"🚀 正在请求接口: {url}")

    try:
        response = requests.post(url, headers=headers, data=payload)

        if response.status_code == 200:
            res_json = response.json()
            data = res_json.get("data", {})

            # 🎯 核心修正：数据在 'event_list' 里
            event_list = data.get("event_list", [])

            if event_list:
                print(f"✅ 成功获取数据！发现 {len(event_list)} 个分类组")

                all_news = []

                # 第一层循环：遍历分类（如“产品进展”、“技术演进”）
                for group in event_list:
                    # 获取分类名称
                    category = group.get("dynamic_group") or group.get(
                        "dynamic_title", "未分类"
                    )

                    # 获取该分类下的新闻列表
                    news_items = group.get("group_list", [])
                    print(f"   📂 处理分类 [{category}]: 包含 {len(news_items)} 条新闻")

                    # 第二层循环：遍历具体新闻
                    for item in news_items:
                        # 处理产品名（有时是列表 ["元宝", "元宝派"]，有时是字符串）
                        product_name = item.get("product_name", "")
                        if isinstance(product_name, list):
                            product_name = ", ".join(product_name)

                        row = {
                            "分类": category,
                            "标题": item.get("title", ""),
                            "摘要/点评": item.get("ai_comment", ""),
                            "涉及产品": product_name,
                            "行业": item.get("industry", ""),
                            "发布时间": item.get("publish_datetime", ""),
                        }
                        all_news.append(row)

                # 导出
                if all_news:
                    df = pd.DataFrame(all_news)
                    filename = f"虎嗅AI日报_{time.strftime('%Y%m%d_%H%M')}.csv"
                    df.to_csv(filename, index=False, encoding="utf-8-sig")
                    print("-" * 30)
                    print(f"🎉 完美！共抓取 {len(all_news)} 条数据")
                    print(df[["分类", "标题"]].head(3))
                    print(f"\n📂 文件已保存: {filename}")
                else:
                    print("⚠️ event_list 不为空，但没提取到具体新闻。")
            else:
                print("⚠️ 返回数据中 'event_list' 为空。")
                print("数据预览:", json.dumps(data, ensure_ascii=False)[:300])
        else:
            print("❌ 请求失败:", response.status_code)

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    scrape_huxiu_success()
