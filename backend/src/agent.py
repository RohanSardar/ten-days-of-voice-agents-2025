import logging

from dotenv import load_dotenv
from typing import List, Optional
import json
from dataclasses import dataclass
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
    AgentTask
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Options to choose from
drinkType = ["latte", "cappuccino", "espresso", "americano", "mocha", "flat white"]
size = ["small", "medium", "large"]
milk = ["whole", "skim", "oat", "none"]
extras = ["sugar", "extra shot", "vanilla syrup", "whipped cream", "none"]

@dataclass
class CoffeeOrder:
    drinkType: str
    size: str
    milk: str
    extras: List[str]
    name: str

class BaristaAssistant(AgentTask[CoffeeOrder]):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""You are a professional coffee shop barista. Your goal is to collect the user's order details: drink type, size, milk, extras, and their name.
            
            Available options:
            - Drinks: {', '.join(drinkType)}
            - Sizes: {', '.join(size)}
            - Milk: {', '.join(milk)}
            - Extras: {', '.join(extras)}
            
            You must strictly validate the user's choices against these options. If a user asks for something not on the list, politely inform them of the available options.
            Collect all information before finalizing the order.
            """
        )
        self._order_data = {}

    @function_tool()
    async def record_drink_type(self, drink_type: str):
        """Record the drink type. Must be one of the available options."""
        drink = drink_type.lower().strip()
        if drink in drinkType:
            self._order_data["drinkType"] = drink
            self._check_completion()
            return f"Recorded drink type: {drink}"
        else:
            return f"Sorry, {drink_type} is not available. We have: {', '.join(drinkType)}"

    @function_tool()
    async def record_size(self, size: str):
        """Record the drink size. Must be one of the available options."""
        size = size.lower().strip()
        if size in size:
            self._order_data["size"] = size
            self._check_completion()
            return f"Recorded size: {size}"
        else:
            return f"Sorry, {size} is not a valid size. We have: {', '.join(size)}"

    @function_tool()
    async def record_milk(self, milk: str):
        """Record the milk choice. Must be one of the available options."""
        milk = milk.lower().strip()
        if milk in milk:
            self._order_data["milk"] = milk
            self._check_completion()
            return f"Recorded milk: {milk}"
        else:
            return f"Sorry, {milk} is not available. We have: {', '.join(milk)}"

    @function_tool()
    async def record_extras(self, extras: List[str]):
        """Record extras. Must be a list of available options."""
        valid_extras = []
        invalid_extras = []
        
        for extra in extras:
            extra = extra.lower().strip()
            if extra in extras:
                valid_extras.append(extra)
            else:
                invalid_extras.append(extra)
        
        if invalid_extras:
            return f"Sorry, we don't have: {', '.join(invalid_extras)}. Available extras: {', '.join(extras)}"
        
        self._order_data["extras"] = valid_extras
        self._check_completion()
        return f"Recorded extras: {', '.join(valid_extras)}"

    @function_tool()
    async def record_name(self, name: str):
        """Record the customer's name."""
        self._order_data["name"] = name
        self._check_completion()
        return f"Recorded name: {name}"

    def _check_completion(self):
        required_keys = {"drinkType", "size", "milk", "extras", "name"}
        if required_keys.issubset(self._order_data.keys()):
            order = CoffeeOrder(
                drinkType=self._order_data["drinkType"],
                size=self._order_data["size"],
                milk=self._order_data["milk"],
                extras=self._order_data["extras"],
                name=self._order_data["name"]
            )
            
            # Save to JSON
            try:
                with open("order.json", "w") as f:
                    json.dump(self._order_data, f, indent=4)
                logger.info("Order saved to order.json")
            except Exception as e:
                logger.error(f"Failed to save order: {e}")

            self.complete(order)

            summary = (f"Order Complete! {order.name}, here is your summary: "
                       f"{order.size} {order.drinkType} with {order.milk} milk "
                       f"and {', '.join(order.extras)}.")
            
            pass


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline 
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear 
        tts=murf.TTS(
                voice="en-IN-arohi", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        preemptive_generation=True,
    )


    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)


    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=BaristaAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
