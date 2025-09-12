import asyncio
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, WorkerOptions, cli
from livekit.plugins import openai, cartesia, deepgram, noise_cancellation, silero
import logging
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

logger.info(f"LIVEKIT_URL: {os.getenv('LIVEKIT_URL')}")
logger.info(f"LIVEKIT_API_KEY: {os.getenv('LIVEKIT_API_KEY')[:4]}... (hidden)")
logger.info(f"DEEPGRAM_API_KEY: {os.getenv('DEEPGRAM_API_KEY')[:4]}... (hidden)")
logger.info(f"CARTESIA_API_KEY: {os.getenv('CARTESIA_API_KEY')[:4]}... (hidden)")
logger.info(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:4]}... (hidden)")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

async def entrypoint(ctx: agents.JobContext):
   
    logger.info(f"Joining room: {ctx.room.name}")
    try:
        logger.info("Initializing Deepgram STT")
        stt = deepgram.STT(model="nova-3", language="multi")
        logger.info("Deepgram STT initialized")

        logger.info("Initializing OpenAI LLM")
        llm = openai.LLM(model="gpt-4o-mini")
        logger.info("OpenAI LLM initialized")

        logger.info("Initializing Cartesia TTS")
        tts = cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02")
        logger.info("Cartesia TTS initialized")

        logger.info("Initializing Silero VAD")
        vad = silero.VAD.load()
        logger.info("Silero VAD initialized")

        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
        )
        logger.info("AgentSession initialized")

        await session.start(
            room=ctx.room,
            agent=Assistant(),
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        logger.info(f"Agent started in room: {ctx.room.name}")
        await ctx.connect()
        logger.info("Agent connected to LiveKit")

        await session.generate_reply(instructions="Greet the user and offer assistance.")
    except Exception as e:
        logger.error(f"Error in entrypoint: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Use the official way to start the worker
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
