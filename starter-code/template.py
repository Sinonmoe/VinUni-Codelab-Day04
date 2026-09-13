"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import sys
import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — Trợ lý Trí tuệ Nhân tạo chính thức phục vụ hệ sinh thái Tập đoàn Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng của hệ sinh thái Vingroup (VinFast, Vinpearl).
- Giọng nói & Phong cách: Chuyên nghiệp, lịch sự, thân thiện, trung thực và chính xác.

## 2. AVAILABLE TOOLS
- search_product_catalog(category: str, max_price: int): Tra cứu sản phẩm ô tô điện (xe_dien) hoặc dịch vụ du lịch (du_lich) của Vingroup theo mức giá tối đa.
- submit_support_ticket(customer_name: str, issue_description: str, priority: str): Ghi nhận phản ánh, sự cố kỹ thuật hoặc yêu cầu hỗ trợ vào hệ thống ticket.

## 3. CORE RULES
1. Tuyệt đối KHÔNG BAO GIỜ tự bịa đặt dữ liệu sản phẩm, giá bán, hay trạng thái phòng/xe.
2. BẮT BUỘC phải gọi tool khi người dùng yêu cầu tra cứu sản phẩm hoặc gửi phản ánh sự cố.
3. Nếu dữ liệu tra cứu rỗng, thông báo trung thực và lịch sự rằng không tìm thấy sản phẩm phù hợp.
4. Luôn phản hồi bằng tiếng Việt chuẩn mực, rõ ràng và mạch lạc.

## 4. OPERATIONAL BOUNDARIES
- Chỉ giải đáp các thắc mắc liên quan đến sản phẩm và dịch vụ thuộc Tập đoàn Vingroup (VinFast, Vinpearl).
- Lịch sự từ chối các câu hỏi nằm ngoài phạm vi hoạt động này.

## 5. OUTPUT CONTRACT
Tuân thủ định dạng suy luận ReAct khi giải quyết vấn đề:
- Thought: Suy nghĩ và phân tích ý định của người dùng.
- Action: Tên công cụ cần thực thi (hoặc None nếu trả lời trực tiếp).
- Action Input: Tham số đầu vào truyền cho công cụ dưới dạng JSON.
- Observation: Dữ liệu thực tế quan sát được sau khi gọi công cụ.
- Final Answer: Câu trả lời tổng hợp cuối cùng gửi tới khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """Trả về câu trả lời baseline (không dùng tool) để quan sát hiện tượng hallucination."""
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intent(self, user_input: str) -> Dict[str, Any]:
        """
        TODO 3: Phân tích intent từ user_input
        - Xác định cần gọi tool nào (catalog? ticket? cả hai? FAQ?)
        - Trích xuất tham số tương ứng cho từng tool.
        """
        text_lower = user_input.lower()

        # Kiểm tra FAQ (các câu hỏi chung về chính sách, bảo hành không kèm lỗi cụ thể hay yêu cầu xem giá)
        is_faq_warranty = any(k in text_lower for k in ["bảo hành", "bao lâu", "chính sách"]) and not any(
            k in text_lower for k in ["lỗi", "hỏng", "sự cố", "khiếu nại", "gấp", "nghiêm trọng"]
        ) and not any(
            k in text_lower for k in ["xem", "tìm", "giá", "dưới", "triệu", "tỷ"]
        )

        # 1. Intent tra cứu catalog
        has_catalog_keyword = any(k in text_lower for k in [
            "xe điện", "xe dien", "vinfast", "resort", "du lịch", "du lich", "vinpearl", "khách sạn", "khach san"
        ])
        has_search_intent = any(k in text_lower for k in [
            "xem", "tìm", "mua", "giá", "dưới", "triệu", "tỷ", "có xe", "chi phí", "bao nhiêu"
        ])
        needs_catalog = (has_catalog_keyword and has_search_intent and not is_faq_warranty)

        # 2. Intent tạo support ticket
        ticket_keywords = [
            "lỗi", "sự cố", "hỏng", "ẩm mốc", "khiếu nại", "phản hồi",
            "nghiêm trọng", "cần xử lý gấp", "hỗ trợ gấp", "ghi nhận"
        ]
        has_ticket_keywords = any(k in text_lower for k in ticket_keywords)
        has_customer_intro = bool(re.search(r'(?:tôi tên|tên tôi là|tôi tên là|tên là)', text_lower))
        needs_ticket = has_ticket_keywords or has_customer_intro

        catalog_args = {}
        if needs_catalog:
            if any(k in text_lower for k in ["du lịch", "du lich", "resort", "vinpearl", "khách sạn", "khach san", "phòng"]):
                category = "du_lich"
            else:
                category = "xe_dien"

            max_price = 999999999999
            match_ty = re.search(r'(\d+(?:[\.,]\d+)?)\s*(?:tỷ|ty\b)', text_lower)
            match_trieu = re.search(r'(\d+(?:[\.,]\d+)?)\s*(?:triệu|trieu|tr\b)', text_lower)
            match_digits = re.search(r'(?:dưới|tối đa|giá|<=|<)\s*(\d{7,})', text_lower)

            if match_ty:
                val = float(match_ty.group(1).replace(',', '.'))
                max_price = int(val * 1_000_000_000)
            elif match_trieu:
                val = float(match_trieu.group(1).replace(',', '.'))
                max_price = int(val * 1_000_000)
            elif match_digits:
                max_price = int(match_digits.group(1))

            catalog_args = {"category": category, "max_price": max_price}

        ticket_args = {}
        if needs_ticket:
            name_match = re.search(
                r'(?:tôi tên là|tên tôi là|tôi tên|tên là)\s*[:]?\s*([^\d,\.:;\?!]+?)(?:,|\.|\bvà\b|\bxe\b|\bphòng\b|\bmức\b|$)',
                user_input,
                re.IGNORECASE
            )
            customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

            if any(k in text_lower for k in ["nghiêm trọng", "khẩn cấp", "gấp", "nguy hiểm", "cao"]):
                priority = "high"
            elif any(k in text_lower for k in ["thấp", "nhẹ"]):
                priority = "low"
            else:
                priority = "medium"

            issue_match = re.search(
                r'((?:xe|phòng|dịch vụ|thiết bị|hệ thống|chuyến đi)[^,\.]*(?:bị|lỗi|hỏng|ẩm mốc|kém|chậm|trục trặc)[^,\.]*)',
                user_input,
                re.IGNORECASE
            )
            if issue_match:
                issue_description = issue_match.group(1).strip()
            else:
                desc_match = re.search(
                    r'(?:phản hồi|khiếu nại|sự cố|lỗi|vấn đề)\s*[:]?\s*([^,\.]+)',
                    user_input,
                    re.IGNORECASE
                )
                issue_description = desc_match.group(1).strip() if desc_match else user_input.strip()

            ticket_args = {
                "customer_name": customer_name,
                "issue_description": issue_description,
                "priority": priority
            }

        faq_answer = ""
        if not needs_catalog and not needs_ticket:
            if "bảo hành" in text_lower or "pin" in text_lower:
                faq_answer = (
                    "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km "
                    "(tùy điều kiện nào đến trước), áp dụng cho các dòng xe điện như VF 5, VF 8, VF 9. "
                    "VinFast cam kết sửa chữa hoặc đổi mới pin miễn phí nếu dung lượng pin khả dụng giảm dưới 70%."
                )
            else:
                faq_answer = (
                    "VinAssistant sẵn sàng hỗ trợ quý khách về các sản phẩm ô tô điện VinFast, "
                    "dịch vụ nghỉ dưỡng Vinpearl và tiếp nhận các yêu cầu bảo hành, kỹ thuật."
                )

        return {
            "needs_catalog": needs_catalog,
            "catalog_args": catalog_args,
            "needs_ticket": needs_ticket,
            "ticket_args": ticket_args,
            "is_faq": (not needs_catalog and not needs_ticket),
            "faq_answer": faq_answer
        }

    def _format_catalog_answer(self, results: List[Dict[str, Any]]) -> str:
        """Định dạng kết quả trả về từ search_product_catalog."""
        if not results or len(results) == 0:
            return "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn."

        lines = [f"Tìm thấy {len(results)} sản phẩm phù hợp:"]
        for p in results:
            name = p.get("name", "Sản phẩm")
            price = p.get("price_vnd", 0)
            desc = p.get("description", "")
            lines.append(f"- {name}: Giá {price:,} VNĐ. {desc}")
        return "\n".join(lines)

    def _format_ticket_answer(self, ticket_info: Dict[str, Any]) -> str:
        """Định dạng kết quả trả về từ submit_support_ticket."""
        if not ticket_info or "ticket_id" not in ticket_info:
            return "Không thể tạo ticket hỗ trợ do xảy ra lỗi hệ thống."
        ticket_id = ticket_info.get("ticket_id", "")
        customer_name = ticket_info.get("customer_name", "Khách hàng")
        return (
            f"Yêu cầu hỗ trợ của quý khách {customer_name} đã được ghi nhận thành công. "
            f"Mã ticket hỗ trợ: {ticket_id}. "
            f"Đội ngũ kỹ thuật và CSKH sẽ liên hệ xử lý trong thời gian sớm nhất."
        )

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        # TODO 3: Phân tích intent từ user_input
        intents = self._detect_intent(user_input)
        actions = []
        if intents["needs_catalog"]:
            actions.append(("search_product_catalog", intents["catalog_args"]))
        if intents["needs_ticket"]:
            actions.append(("submit_support_ticket", intents["ticket_args"]))

        iteration = 1
        executed_results = {}

        # TODO 4: Xây dựng Agent Loop (while iteration <= self.max_iterations)
        while iteration <= self.max_iterations:
            # FAQ case: Trả lời trực tiếp ngay tại Iteration 1
            if not actions:
                faq_answer = intents.get("faq_answer", "")
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Câu hỏi FAQ/chính sách, không cần gọi công cụ bên ngoài.",
                    "action": None,
                    "action_input": {},
                    "observation": faq_answer
                })
                return {
                    "answer": faq_answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            # Single tool case: Hoàn thành trong 1 iteration
            if len(actions) == 1:
                tool_name, tool_args = actions[0]
                tool_fn = TOOL_MAP.get(tool_name)
                result = tool_fn(**tool_args) if tool_fn else None

                self.trace.append({
                    "iteration": iteration,
                    "thought": f"Yêu cầu cần gọi công cụ {tool_name} với tham số {tool_args}.",
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": result
                })

                if tool_name == "search_product_catalog":
                    answer = self._format_catalog_answer(result)
                else:
                    answer = self._format_ticket_answer(result)

                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            # Multi-tool case (ví dụ parallel tool calling):
            # Iteration 1: Gọi tool #1
            # Iteration 2: Gọi tool #2
            # Iteration 3: Tổng hợp Final Answer từ trace
            if iteration == 1:
                tool_name, tool_args = actions[0]
                tool_fn = TOOL_MAP.get(tool_name)
                res = tool_fn(**tool_args) if tool_fn else None
                executed_results[tool_name] = res

                self.trace.append({
                    "iteration": iteration,
                    "thought": f"Bước 1: Gọi {tool_name} để lấy dữ liệu.",
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": res
                })
                iteration += 1
                continue

            elif iteration == 2:
                tool_name, tool_args = actions[1]
                tool_fn = TOOL_MAP.get(tool_name)
                res = tool_fn(**tool_args) if tool_fn else None
                executed_results[tool_name] = res

                self.trace.append({
                    "iteration": iteration,
                    "thought": f"Bước 2: Gọi {tool_name} để ghi nhận thông tin.",
                    "action": tool_name,
                    "action_input": tool_args,
                    "observation": res
                })
                iteration += 1
                continue

            elif iteration >= 3:
                catalog_res = executed_results.get("search_product_catalog", [])
                ticket_res = executed_results.get("submit_support_ticket", {})

                cat_text = self._format_catalog_answer(catalog_res)
                tick_text = self._format_ticket_answer(ticket_res)
                final_answer = f"{cat_text}\n\n{tick_text}"

                self.trace.append({
                    "iteration": iteration,
                    "thought": "Đã thu thập đầy đủ kết quả từ các bước. Tổng hợp câu trả lời cuối cùng.",
                    "action": "final_answer",
                    "action_input": {},
                    "observation": final_answer
                })
                return {
                    "answer": final_answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

        # Max iterations guard
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": iteration - 1,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
